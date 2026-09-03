"""AutoProbe case workspace — Living17 (SubpopBench). SELF-CONTAINED.

Runs as `python train.py` with no arguments, per the AutoProbe workspace
contract. Exits 0 on success, non-zero with a traceback on failure.

This is a faithful epoch-structured rewrite of SubpopBench's step-based
`subpopbench/train.py`. The training semantics (hparams, loaders, algorithm
construction, update call) are unchanged so results stay comparable to the
published tables; the loop is reshaped into epochs so a probe has a natural
per-epoch hook, and the tensorboard/checkpoint plumbing is dropped.

Nothing here is probe-aware. Stage 3 of the pipeline adds `prober.py` and the
`record(...)` / `conclude(...)` calls.

WHY THIS FILE IS LARGE
----------------------
Everything an improvement loop is allowed to change lives HERE, inlined
verbatim from `subpopbench/` instead of imported:

  * hyper-parameter defaults   (was subpopbench/hparams_registry.py)
  * optimizers                 (was subpopbench/learning/optimizers.py)
  * networks / featurizers     (was subpopbench/models/networks.py,
                                    subpopbench/models/wide_resnet.py)
  * joint-DRO robust loss      (was subpopbench/learning/joint_dro.py)
  * ALL algorithms, ERM..LISA  (was subpopbench/learning/algorithms.py)

Two reasons. First, an agent told to "switch to GroupDRO" or "add an LR
schedule" can do it in the one file it already has open, and can read the
exact code that will run instead of chasing it across a package.

Second, and the reason that actually bites: AutoProbe's snapshot git tracks
ONLY train.py, and `_maybe_revert_on_regression` restores ONLY train.py. An
edit that landed in `subpopbench/learning/algorithms.py` used to SURVIVE its
own revert — the orchestrator would record "reverted" while the change stayed
live on disk, leaving the workspace disagreeing with its own history. With
the code here, keep/revert is honest and a round is reproducible from its
snapshot alone.

WHAT IS DELIBERATELY STILL IMPORTED
-----------------------------------
The measurement side stays in `subpopbench/`, untouched and frozen:

  * `subpopbench.dataset.datasets` — builds Living17, including its
    subpopulation structure. This DEFINES the task; editing it changes the
    problem instead of solving it.
  * `subpopbench.dataset.fast_dataloader`
  * `subpopbench.utils.eval_helper` — `eval_metrics` is THE scorer. Kept
    external so its numbers stay definitionally identical to the published
    tables and can be re-run and audited independently of this file.

Do not inline or edit those to make a metric move.
"""
import copy
import hashlib
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn.init as init
import torchvision.models
import timm
from torch.hub import load_state_dict_from_url
from torch.optim import AdamW
from transformers import BertModel, DistilBertModel, GPT2Model, AutoModel
from transformers import get_scheduler

# FROZEN measurement side — see module docstring. Do not inline or edit these.
from subpopbench.dataset import datasets
from subpopbench.dataset.fast_dataloader import InfiniteDataLoader, FastDataLoader
from subpopbench.utils import eval_helper


# ==============================================================================
# HYPERPARAMETER REGISTRY
# Inlined verbatim from subpopbench/hparams_registry.py
# ==============================================================================

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)




def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    IMAGE_DATASETS = ["Waterbirds", "CelebA", "MetaShift", "ImagenetBG", "NICOpp",
                      "MIMICNoFinding", "CXRMultisite", "CheXpertNoFinding",
                      "Living17", "Entity13", "Entity30", "Nonliving26", "CMNIST"]
    TEXT_DATASETS = ["CivilCommentsFine", "MultiNLI", "CivilComments"]
    TABULAR_DATASET = ["MIMICNotes"]

    HALF_BS_ALGOS = ['LfF']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert name not in hparams
        random_state = np.random.RandomState(
            seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions

    _hparam('resnet18', False, lambda r: False)
    # nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False, lambda r: bool(r.choice([False, False])))

    if algorithm in ['ReSample', 'CRT']:
        _hparam('group_balanced', True, lambda r: True)
    else:
        _hparam('group_balanced', False, lambda r: False)

    # Algorithm-specific hparam definitions
    # Each block of code below corresponds to one algorithm

    if algorithm == 'CBLoss':
        _hparam('beta', 0.9999, lambda r: 1 - 10**r.uniform(-5, -2))

    elif algorithm == 'Focal':
        _hparam('gamma', 1, lambda r: 0.5 * 10**r.uniform(0, 1))

    elif algorithm == 'LDAM':
        _hparam('max_m', 0.5, lambda r: 10**r.uniform(-1, -0.1))
        _hparam('scale', 30., lambda r: r.choice([10., 30.]))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500, lambda r: int(10**r.uniform(0, 4)))

    elif "Mixup" in algorithm:
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, 1))

    elif "GroupDRO" in algorithm:
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))

    elif algorithm in ["MMD", "CORAL"]:
        _hparam('mmd_gamma', 1., lambda r: 10**r.uniform(-1, 1))

    elif 'CRT' in algorithm:
        _hparam('stage1_model', 'model.pkl', lambda r: 'model.pkl')

    elif algorithm == 'CVaRDRO':
        _hparam('joint_dro_alpha', 0.1, lambda r: 10**r.uniform(-2, 0))

    elif algorithm == 'JTT':
        _hparam('first_stage_step_frac', 0.5, lambda r: r.uniform(0.2, 0.8))
        _hparam('jtt_lambda', 10, lambda r: 10**r.uniform(0, 2.5))

    elif algorithm == 'LfF':
        _hparam('LfF_q', 0.7, lambda r: r.uniform(0.05, 0.95))

    elif algorithm == 'LISA':
        _hparam('LISA_alpha', 2., lambda r: 10**r.uniform(-1, 1))
        _hparam('LISA_p_sel', 0.5, lambda r: r.uniform(0, 1))
        _hparam('LISA_mixup_method', 'mixup', lambda r: r.choice(['mixup', 'cutmix']))

    elif algorithm == 'DFR':
        _hparam('stage1_model', 'model.pkl', lambda r: 'model.pkl')
        _hparam('dfr_reg', .1, lambda r: 10**r.uniform(-2, 0.5))

    # Dataset-and-algorithm-specific hparam definitions
    # Each block of code below corresponds to exactly one hparam. Avoid nested conditionals

    if dataset in {"Living17", "Entity13", "Entity30", "Nonliving26"}:
        _hparam('pretrained', False, lambda r: False)
    else:
        _hparam('pretrained', True, lambda r: True)

    if dataset in TABULAR_DATASET:
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    if dataset in IMAGE_DATASETS + TABULAR_DATASET:
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4, -2))
    else:
        _hparam('lr', 1e-5, lambda r: 10**r.uniform(-5.5, -4))

    _hparam('weight_decay', 1e-4, lambda r: 10**r.uniform(-6, -3))

    if dataset in TEXT_DATASETS:
        _hparam('optimizer', 'adamw', lambda r: 'adamw')
    else:
        _hparam('optimizer', 'sgd', lambda r: 'sgd')

    if dataset in TEXT_DATASETS:
        _hparam('last_layer_dropout', 0.5, lambda r: r.choice([0., 0.1, 0.5]))
    else:
        _hparam('last_layer_dropout', 0., lambda r: 0.)

    if algorithm in HALF_BS_ALGOS:
        if dataset in TEXT_DATASETS:
            _hparam('batch_size', 16, lambda r: int(2**r.uniform(3, 4)))
        elif dataset in TABULAR_DATASET:
            _hparam('batch_size', 128, lambda r: int(2 ** r.uniform(7, 9)))
        else:
            _hparam('batch_size', 54, lambda r: int(2**r.uniform(5, 5.75)))
    else:
        if dataset in TEXT_DATASETS:
            _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5.5)))
        elif dataset in TABULAR_DATASET:
            _hparam('batch_size', 256, lambda r: int(2 ** r.uniform(7, 10)))
        else:
            _hparam('batch_size', 108, lambda r: int(2**r.uniform(6, 6.75)))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}


# ==============================================================================
# OPTIMIZERS
# Inlined verbatim from subpopbench/learning/optimizers.py
# ==============================================================================



def get_bert_optim(network, lr, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    no_decay_params = []
    for n, p in network.named_parameters():
        if any(nd in n for nd in no_decay):
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8)
    return optimizer


def get_sgd_optim(network, lr, weight_decay):
    return torch.optim.SGD(
        network.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9)


get_optimizers = {
    "sgd": get_sgd_optim,
    "adamw": get_bert_optim
}


# ==============================================================================
# WIDE RESNET
# Inlined verbatim from subpopbench/models/wide_resnet.py
# ==============================================================================


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1, stride=stride,
                    bias=True), )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    """WideResNet with the softmax layer chopped off"""
    def __init__(self, input_shape, depth, widen_factor, dropout_rate):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        # print(' | WideResNet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(input_shape[0], nStages[0])
        self.layer1 = self._wide_layer(
            WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(
            WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(
            WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)

        self.n_outputs = nStages[3]

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        return out[:, :, 0, 0]


# ==============================================================================
# NETWORKS / FEATURIZERS
# Inlined verbatim from subpopbench/models/networks.py
# ==============================================================================


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
                                      for _ in range(hparams['mlp_depth'] - 2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class PretrainedImageModel(torch.nn.Module):

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters."""
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class ResNet(PretrainedImageModel):

    def __init__(self, input_shape, hparams, pretrained=True, freeze_bn=False):
        super(ResNet, self).__init__()

        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=pretrained)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=pretrained)
            self.n_outputs = 2048

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['last_layer_dropout'])

        if freeze_bn:
            self.freeze_bn()
        else:
            assert hparams['last_layer_dropout'] == 0.


class TimmModel(PretrainedImageModel):

    def __init__(self, name, input_shape, hparams, pretrained=True, freeze_bn=False):
        super().__init__()

        self.network = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.n_outputs = self.network.num_features
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['last_layer_dropout'])

        if freeze_bn:
            self.freeze_bn()
        else:
            assert hparams['last_layer_dropout'] == 0.


class HubModel(PretrainedImageModel):

    def __init__(self, name1, name2, input_shape, hparams, pretrained=True, freeze_bn=False):
        super().__init__()

        self.network = torch.hub.load(name1, name2, force_reload=True)
        if hasattr(self.network, 'num_features'):
            self.n_outputs = self.network.num_features
        else:
            self.n_outputs = 2048
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['last_layer_dropout'])

        if freeze_bn:
            self.freeze_bn()
        else:
            assert hparams['last_layer_dropout'] == 0.


class ImportedModel(PretrainedImageModel):

    def __init__(self, network, n_outputs, input_shape, hparams, pretrained=True, freeze_bn=False):
        super().__init__()

        self.network = network
        self.n_outputs = n_outputs
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['last_layer_dropout'])

        if freeze_bn:
            self.freeze_bn()
        else:
            assert hparams['last_layer_dropout'] == 0.


class MNIST_CNN(nn.Module):

    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        # x = F.normalize(x, dim=1)
        return x


class BertFeatureWrapper(torch.nn.Module):

    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        self.n_outputs = model.config.hidden_size
        classifier_dropout = (
            hparams['last_layer_dropout'] if hparams['last_layer_dropout'] != 0. else model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, x):
        kwargs = {
            'input_ids': x[:, :, 0],
            'attention_mask': x[:, :, 1]
        }
        if x.shape[-1] == 3:
            kwargs['token_type_ids'] = x[:, :, 2]
        output = self.model(**kwargs)
        if hasattr(output, 'pooler_output'):
            return self.dropout(output.pooler_output)
        else:
            return self.dropout(output.last_hidden_state[:, 0, :])


def replace_module_prefix(state_dict, prefix, replace_with=""):
    state_dict = {
        (key.replace(prefix, replace_with, 1)
         if key.startswith(prefix) else key): val
        for (key, val) in state_dict.items()
    }
    return state_dict


def get_torchvision_state_dict(url):
    model = load_state_dict_from_url(url)
    model_trunk = model["classy_state_dict"]["base_model"]["model"]["trunk"] if 'classy_state_dict' in model else model
    return replace_module_prefix(model_trunk, "_feature_blocks.")


def imagenet_resnet50_ssl(URL):
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Identity()
    model.load_state_dict(get_torchvision_state_dict(URL))
    model.fc.in_features = 2048
    model.n_outputs = 2048
    return model


def load_swag(URL):
    m = torchvision.models.vit_b_16(pretrained=False)
    m.heads = torch.nn.Identity()    
    state_dict = load_state_dict_from_url(URL)
    state_dict_new = {}
    for (key, val) in state_dict.items():
        if 'layer_' in key:
            key = key.replace('layer_', 'encoder_layer_', 1)
        if key == 'encoder.pos_embedding':
            val = val.permute((1, 0, 2))        
        state_dict_new[key] = val 
    m.load_state_dict(state_dict_new)
    m.n_outputs = 768
    return m


SIMCLR_RN50_URL = "https://dl.fbaipublicfiles.com/vissl/model_zoo/" \
                  "simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch"
BARLOWTWINS_RN50_URL = "https://dl.fbaipublicfiles.com/vissl/model_zoo/" \
                       "barlow_twins/barlow_twins_32gpus_4node_imagenet1k_1000ep_resnet50.torch"


def Featurizer(data_type, input_shape, hparams):
    """Auto-select an appropriate featurizer for the given data type & input shape."""
    if data_type == "images":
        if len(input_shape) == 1:
            return MLP(input_shape[0], hparams["mlp_width"], hparams)
        elif input_shape[1:3] == (28, 28):
            return MNIST_CNN(input_shape)
        elif input_shape[1:3] == (32, 32):
            return WideResNet(input_shape, 16, 2, 0.)
        elif input_shape[1:3] == (224, 224):
            if hparams['image_arch'] == 'resnet_sup_in1k':
                return ResNet(input_shape, hparams, hparams['pretrained'])
            elif hparams['image_arch'] in ['vit_sup_in1k', 'vit_sup_in21k', 'vit_clip_oai',
                                           'vit_clip_laion', 'resnet_sup_in21k', 'vit_dino_in1k']:
                return TimmModel({
                    'resnet_sup_in21k': 'tresnet_m_miil_in21k',  # https://github.com/Alibaba-MIIL/ImageNet21K
                    'vit_sup_in1k': 'vit_base_patch32_224.augreg_in1k',  # https://arxiv.org/abs/2106.10270
                    'vit_sup_in21k': 'vit_base_patch32_224.augreg_in21k',
                    'vit_clip_oai': 'vit_base_patch32_clip_224.openai',
                    'vit_clip_laion': 'vit_base_patch32_clip_224.laion2b',
                    'vit_dino_in1k': 'vit_base_patch16_224.dino'  # https://github.com/facebookresearch/dino
                }[hparams['image_arch']], input_shape, hparams, hparams['pretrained'])
            elif hparams['image_arch'] == 'resnet_dino_in1k':
                return ImportedModel(
                    imagenet_resnet50_ssl(
                        'https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth'),
                    2048, input_shape, hparams, hparams['pretrained']
                )
            elif hparams['image_arch'] == 'vit_sup_swag':
                # https://github.com/facebookresearch/SWAG
                return ImportedModel(load_swag('https://dl.fbaipublicfiles.com/SWAG/vit_b16.torch'),
                                     768, input_shape, hparams, hparams['pretrained'])
            elif hparams['image_arch'] in ['resnet_barlow_in1k', 'resnet_simclr_in1k']:                
                return ImportedModel(imagenet_resnet50_ssl({
                    'resnet_simclr_in1k': SIMCLR_RN50_URL,
                    'resnet_barlow_in1k': BARLOWTWINS_RN50_URL
                }[hparams['image_arch']]), 2048, input_shape, hparams, hparams['pretrained'])
        else:
            raise NotImplementedError
    elif data_type == "text":
        if hparams['text_arch'] == 'bert-base-uncased':
            text_model = BertModel.from_pretrained(hparams['text_arch'])
        elif hparams['text_arch'] in ['xlm-roberta-base', 'allenai/scibert_scivocab_uncased']:
            text_model = AutoModel.from_pretrained(hparams['text_arch'])
        elif hparams['text_arch'] == 'gpt2':
            text_model = GPT2Model.from_pretrained('gpt2')
        elif hparams['text_arch'] == 'distilbert-base-uncased':
            text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        else:
            raise NotImplementedError
        return BertFeatureWrapper(text_model, hparams)
    elif data_type == "tabular":
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    else:
        raise NotImplementedError(f"{data_type} not supported.")


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


# ==============================================================================
# JOINT DRO ROBUST LOSS
# Inlined verbatim from subpopbench/learning/joint_dro.py
# ==============================================================================


GEOMETRIES = ('cvar')
MIN_REL_DIFFERENCE = 1e-5


def cvar_value(p, v, reg):
    """Returns <p, v> - reg * KL(p, uniform) for Torch tensors"""
    m = p.shape[0]

    with torch.no_grad():
        idx = torch.nonzero(p)  # where is annoyingly backwards incompatible
        kl = np.log(m) + (p[idx] * torch.log(p[idx])).sum()

    return torch.dot(p, v) - reg * kl


def bisection(eta_min, eta_max, f, tol=1e-6, max_iter=500):
    """Expects f an increasing function and return eta in [eta_min, eta_max] 
    s.t. |f(eta)| <= tol (or the best solution after max_iter iterations"""
    lower = f(eta_min)
    upper = f(eta_max)

    # until the root is between eta_min and eta_max, double the length of the 
    # interval starting at either endpoint.
    while lower > 0 or upper < 0:
        length = eta_max - eta_min
        if lower > 0:
            eta_max = eta_min
            eta_min = eta_min - 2 * length
        if upper < 0:
            eta_min = eta_max
            eta_max = eta_max + 2 * length

        lower = f(eta_min)
        upper = f(eta_max)

    for _ in range(max_iter):
        eta = 0.5 * (eta_min + eta_max)

        v = f(eta)

        if torch.abs(v) <= tol:
            return eta

        if v > 0:
            eta_max = eta
        elif v < 0:
            eta_min = eta

    return 0.5 * (eta_min + eta_max)


class RobustLoss(torch.nn.Module):
    """PyTorch module for the batch robust loss estimator"""
    def __init__(self, size, reg, geometry, tol=1e-4, max_iter=1000, debugging=False):
        r"""
        Parameters
        ----------
        size : float
            Size of the uncertainty set (\rho for \chi^2 and \alpha for CVaR)
            Set float('inf') for unconstrained
        reg : float
            Strength of the regularizer, entropy if geometry == 'cvar'
            $\chi^2$ divergence if geometry == 'chi-square'
        geometry : string
            Element of GEOMETRIES
        tol : float, optional
            Tolerance parameter for the bisection
        max_iter : int, optional
            Number of iterations after which to break the bisection
        """
        super().__init__()
        self.size = size
        self.reg = reg
        self.geometry = geometry
        self.tol = tol
        self.max_iter = max_iter
        self.debugging = debugging

        self.is_erm = size == 0

        if geometry not in GEOMETRIES:
            raise ValueError('Geometry %s not supported' % geometry)

        if geometry == 'cvar' and self.size > 1:
            raise ValueError(f'alpha should be < 1 for cvar, is {self.size}')

    def best_response(self, v):
        size = self.size
        reg = self.reg
        m = v.shape[0]

        if self.geometry == 'cvar':
            if self.reg > 0:
                if size == 1.0:
                    return torch.ones_like(v) / m

                def p(eta):
                    x = (v - eta) / reg
                    return torch.min(torch.exp(x),
                                     torch.Tensor([1 / size]).type(x.dtype)) / m

                def bisection_target(eta):
                    return 1.0 - p(eta).sum()

                eta_min = reg * torch.logsumexp(v / reg - np.log(m), 0)
                eta_max = v.max()

                if torch.abs(bisection_target(eta_min)) <= self.tol:
                    return p(eta_min)
            else:
                cutoff = int(size * m)
                surplus = 1.0 - cutoff / (size * m)

                p = torch.zeros_like(v)
                idx = torch.argsort(v, descending=True)
                p[idx[:cutoff]] = 1.0 / (size * m)
                if cutoff < m:
                    p[idx[cutoff]] = surplus
                return p

        if self.geometry == 'chi-square':
            if (v.max() - v.min()) / v.max() <= MIN_REL_DIFFERENCE:
                return torch.ones_like(v) / m

            if size == float('inf'):
                assert reg > 0

                def p(eta):
                    return torch.relu(v - eta) / (reg * m)

                def bisection_target(eta):
                    return 1.0 - p(eta).sum()

                eta_min = min(v.sum() - reg * m, v.min())
                eta_max = v.max()

            else:
                assert size < float('inf')

                # failsafe for batch sizes small compared to
                # uncertainty set size
                if m <= 1 + 2 * size:
                    out = (v == v.max()).float()
                    out /= out.sum()
                    return out

                if reg == 0:
                    def p(eta):
                        pp = torch.relu(v - eta)
                        return pp / pp.sum()

                    def bisection_target(eta):
                        pp = p(eta)
                        w = m * pp - torch.ones_like(pp)
                        return 0.5 * torch.mean(w ** 2) - size

                    eta_min = -(1.0 / (np.sqrt(2 * size + 1) - 1)) * v.max()
                    eta_max = v.max()
                else:
                    def p(eta):
                        pp = torch.relu(v - eta)

                        opt_lam = max(
                            reg, torch.norm(pp) / np.sqrt(m * (1 + 2 * size))
                        )

                        return pp / (m * opt_lam)

                    def bisection_target(eta):
                        return 1 - p(eta).sum()

                    eta_min = v.min() - 1
                    eta_max = v.max()

        eta_star = bisection(
            eta_min, eta_max, bisection_target,
            tol=self.tol, max_iter=self.max_iter)

        if self.debugging:
            return p(eta_star), eta_star
        return p(eta_star)

    def forward(self, v):
        """Value of the robust loss
        Note that the best response is computed without gradients
        Parameters
        ----------
        v : torch.Tensor
            Tensor containing the individual losses on the batch of examples
        Returns
        -------
        loss : torch.float
            Value of the robust loss on the batch of examples
        """
        if self.is_erm:
            return v.mean()
        else:
            with torch.no_grad():
                p = self.best_response(v)

            if self.geometry == 'cvar':
                return cvar_value(p, v, self.reg)


# ==============================================================================
# MIXUP HELPER
# Inlined verbatim from subpopbench/utils/misc.py
# ==============================================================================

def mixup_data(x, y, alpha=1., device="cpu"):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


# ==============================================================================
# ALGORITHMS
# Inlined verbatim from subpopbench/learning/algorithms.py
# ==============================================================================


ALGORITHMS = [
    'ERM',
    # subgroup methods
    'GroupDRO',
    'IRM',
    'CVaRDRO',
    'JTT',
    'LfF',
    'LISA',
    'DFR',
    # data augmentation
    'Mixup',
    # domain generalization methods
    'MMD',
    'CORAL',
    # imbalanced learning methods
    'ReSample',
    'ReWeight',
    'SqrtReWeight',
    'CBLoss',
    'Focal',
    'LDAM',
    'BSoftmax',
    'CRT',
    'ReWeightCRT',
    'VanillaCRT'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a subgroup robustness algorithm.
    Subclasses should implement the following:
    - _init_model()
    - _compute_loss()
    - update()
    - return_feats()
    - predict()
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.data_type = data_type
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.num_examples = num_examples

    def _init_model(self):
        raise NotImplementedError

    def _compute_loss(self, i, x, y, a, step):
        raise NotImplementedError

    def update(self, minibatch, step):
        """Perform one update step."""
        raise NotImplementedError

    def return_feats(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def return_groups(self, y, a):
        """Given a list of (y, a) tuples, return indexes of samples belonging to each subgroup"""
        idx_g, idx_samples = [], []
        all_g = y * self.num_attributes + a

        for g in all_g.unique():
            idx_g.append(g)
            idx_samples.append(all_g == g)

        return zip(idx_g, idx_samples)

    @staticmethod
    def return_attributes(all_a):
        """Given a list of attributes, return indexes of samples belonging to each attribute"""
        idx_a, idx_samples = [], []

        for a in all_a.unique():
            idx_a.append(a)
            idx_samples.append(all_a == a)

        return zip(idx_a, idx_samples)


class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(ERM, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

        self.featurizer = Featurizer(data_type, input_shape, self.hparams)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier']
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self._init_model()

    def _init_model(self):
        self.clip_grad = (self.data_type == "text" and self.hparams["optimizer"] == "adamw")

        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers['sgd'](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = None
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        elif self.data_type == "text":
            self.network.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")

    def _compute_loss(self, i, x, y, a, step):
        return self.loss(self.predict(x), y).mean()

    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch
        loss = self._compute_loss(all_i, all_x, all_y, all_a, step)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.data_type == "text":
            self.network.zero_grad()

        return {'loss': loss.item()}

    def return_feats(self, x):
        return self.featurizer(x)

    def predict(self, x):
        return self.network(x)


class GroupDRO(ERM):
    """
    Group DRO minimizes the error at the worst group [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(GroupDRO, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        self.register_buffer(
            "q", torch.ones(self.num_classes * self.num_attributes).cuda())

    def _compute_loss(self, i, x, y, a, step):
        losses = self.loss(self.predict(x), y)

        for idx_g, idx_samples in self.return_groups(y, a):
            self.q[idx_g] *= (self.hparams["groupdro_eta"] * losses[idx_samples].mean()).exp().item()

        self.q /= self.q.sum()

        loss_value = 0
        for idx_g, idx_samples in self.return_groups(y, a):
            loss_value += self.q[idx_g] * losses[idx_samples].mean()

        return loss_value


class ReSample(ERM):
    """Naive resample, with no changes to ERM, but enable balanced sampling in hparams"""


class ReWeight(ERM):
    """Naive inverse re-weighting"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(ReWeight, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        assert len(grp_sizes) == num_classes * num_attributes
        grp_sizes = [x if x else np.inf for x in grp_sizes]
        per_grp_weights = 1 / np.array(grp_sizes)
        per_grp_weights = per_grp_weights / np.sum(per_grp_weights) * len(grp_sizes)
        self.weights_per_grp = torch.FloatTensor(per_grp_weights)

    def _compute_loss(self, i, x, y, a, step):
        losses = self.loss(self.predict(x), y)

        all_g = y * self.num_attributes + a
        loss_value = (self.weights_per_grp.type_as(losses)[all_g] * losses).mean()

        return loss_value


class SqrtReWeight(ReWeight):
    """Square-root inverse re-weighting"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(SqrtReWeight, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        assert len(grp_sizes) == num_classes * num_attributes
        grp_sizes = [x if x else np.inf for x in grp_sizes]
        per_grp_weights = 1 / np.sqrt(np.array(grp_sizes))
        per_grp_weights = per_grp_weights / np.sum(per_grp_weights) * len(grp_sizes)
        self.weights_per_grp = torch.FloatTensor(per_grp_weights)


class CBLoss(ReWeight):
    """Class-balanced loss, https://arxiv.org/pdf/1901.05555.pdf"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(CBLoss, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

        assert len(grp_sizes) == num_classes * num_attributes
        grp_sizes = [x if x else np.inf for x in grp_sizes]
        effective_num = 1. - np.power(self.hparams["beta"], grp_sizes)
        effective_num = np.array(effective_num)
        effective_num[effective_num == 1] = np.inf
        per_grp_weights = (1. - self.hparams["beta"]) / effective_num
        per_grp_weights = per_grp_weights / np.sum(per_grp_weights) * len(grp_sizes)
        self.weights_per_grp = torch.FloatTensor(per_grp_weights)


class Focal(ERM):
    """Focal loss, https://arxiv.org/abs/1708.02002"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(Focal, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

    @staticmethod
    def focal_loss(input_values, gamma):
        p = torch.exp(-input_values)
        loss = (1 - p) ** gamma * input_values
        return loss.mean()

    def _compute_loss(self, i, x, y, a, step):
        return self.focal_loss(self.loss(self.predict(x), y), self.hparams["gamma"])


class LDAM(ERM):
    """LDAM loss, https://arxiv.org/abs/1906.07413"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(LDAM, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        assert len(grp_sizes) == num_classes * num_attributes
        # attribute-agnostic as modifying class-dependent margins
        class_sizes = [np.sum(grp_sizes[i * num_attributes:(i+1) * num_attributes]) for i in range(num_classes)]
        class_sizes = [x if x else np.inf for x in class_sizes]
        m_list = 1. / np.sqrt(np.sqrt(np.array(class_sizes)))
        m_list = m_list * (self.hparams["max_m"] / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list)

    def _compute_loss(self, i, x, y, a, step):
        x = self.predict(x)
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, y.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :].type_as(x), index_float.transpose(0, 1).type_as(x))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        loss_value = F.cross_entropy(self.hparams["scale"] * output, y)

        return loss_value


class BSoftmax(ERM):
    """Balanced softmax, https://arxiv.org/abs/2007.10740"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(BSoftmax, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        assert len(grp_sizes) == num_classes * num_attributes
        # attribute-agnostic as modifying class-dependent margins
        class_sizes = [np.sum(grp_sizes[i * num_attributes:(i+1) * num_attributes]) for i in range(num_classes)]
        self.n_samples_per_cls = torch.FloatTensor(class_sizes)

    def _compute_loss(self, i, x, y, a, step):
        x = self.predict(x)
        spc = self.n_samples_per_cls.type_as(x)
        spc = spc.unsqueeze(0).expand(x.shape[0], -1)
        x = x + spc.log()
        loss_value = F.cross_entropy(input=x, target=y)

        return loss_value


class CRT(ERM):
    """Classifier re-training with balanced sampling during the second earning stage"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(CRT, self).__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        # fix stage 1 trained featurizer
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = False
        # only optimize the classifier
        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers['sgd'](
                self.classifier,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = None
        elif self.data_type == "text":
            self.network.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.classifier,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")


class ReWeightCRT(ReWeight):
    """Classifier re-training with balanced re-weighting during the second earning stage"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(ReWeightCRT, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        # fix stage 1 trained featurizer
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = False
        # only optimize the classifier
        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers['sgd'](
                self.classifier,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = None
        elif self.data_type == "text":
            self.network.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.classifier,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")


class VanillaCRT(ERM):
    """Classifier re-training with normal (instance-balanced) sampling"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(VanillaCRT, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        # fix stage 1 trained featurizer
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = False
        # only optimize the classifier
        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers['sgd'](
                self.classifier,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = None
        elif self.data_type == "text":
            self.network.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.classifier,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")


class DFR(ERM):
    """
    Classifier re-training with sub-sampled, group-balanced, held-out(validation) data and l1 regularization.
    Note that when attribute is unavailable in validation data, group-balanced reduces to class-balanced.
    https://openreview.net/pdf?id=Zb6c8A-Fghk
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(DFR, self).__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        # fix stage 1 trained featurizer
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = False
        # only optimize the classifier
        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers['sgd'](
                self.classifier,
                self.hparams['lr'],
                0.
            )
            self.lr_scheduler = None
        elif self.data_type == "text":
            self.network.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.classifier,
                self.hparams['lr'],
                0.
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")

    def _compute_loss(self, i, x, y, a, step):
        return self.loss(self.predict(x), y).mean() + self.hparams['dfr_reg'] * torch.norm(self.classifier.weight, 1)


class IRM(ERM):
    """Invariant Risk Minimization"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(IRM, self).__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def _compute_loss(self, i, x, y, a, step):
        penalty_weight = self.hparams['irm_lambda'] \
            if self.update_count >= self.hparams['irm_penalty_anneal_iters'] else 1.0
        nll = 0.
        penalty = 0.

        logits = self.network(x)
        for idx_a, idx_samples in self.return_attributes(a):
            nll += F.cross_entropy(logits[idx_samples], y[idx_samples])
            penalty += self._irm_penalty(logits[idx_samples], y[idx_samples])
        nll /= len(a.unique())
        penalty /= len(a.unique())
        loss_value = nll + (penalty_weight * penalty)

        self.update_count += 1
        return loss_value


class Mixup(ERM):
    """Mixup of minibatch data"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(Mixup, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

    def _compute_loss(self, i, x, y, a, step):
        if self.data_type == "text":
            feats = self.featurizer(x)
            feats, yi, yj, lam = mixup_data(feats, y, self.hparams["mixup_alpha"], device="cuda")
            predictions = self.classifier(feats)
        else:
            x, yi, yj, lam = mixup_data(x, y, self.hparams["mixup_alpha"], device="cuda")
            predictions = self.predict(x)
        loss_value = lam * F.cross_entropy(predictions, yi) + (1 - lam) * F.cross_entropy(predictions, yj)
        return loss_value


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions using MMD
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams,
                 grp_sizes=None, gaussian=False):
        super(AbstractMMD, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    @staticmethod
    def my_cdist(x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def _compute_loss(self, i, x, y, a, step):
        all_feats = self.featurizer(x)
        outputs = self.classifier(all_feats)
        objective = F.cross_entropy(outputs, y)

        features = []
        for _, idx_samples in self.return_attributes(a):
            features.append(all_feats[idx_samples])

        penalty = 0.
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                penalty += self.mmd(features[i], features[j])

        if len(features) > 1:
            penalty /= (len(features) * (len(features) - 1) / 2)

        loss_value = objective + (self.hparams['mmd_gamma'] * penalty)
        return loss_value


class MMD(AbstractMMD):
    """MMD using Gaussian kernel"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(MMD, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes, gaussian=True)


class CORAL(AbstractMMD):
    """MMD using mean and covariance difference"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(CORAL, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes, gaussian=False)


class CVaRDRO(ERM):
    """
    DRO with CVaR uncertainty set
    https://arxiv.org/pdf/2010.05893.pdf
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(CVaRDRO, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        self._joint_dro_loss_computer = RobustLoss(hparams['joint_dro_alpha'], 0, "cvar")

    def _compute_loss(self, i, x, y, a, step):
        per_sample_losses = self.loss(self.predict(x), y)
        actual_loss = self._joint_dro_loss_computer(per_sample_losses)
        return actual_loss


class AbstractTwoStage(Algorithm):
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super().__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

        self.stage1_model = ERM(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        self.first_stage_step_frac = hparams['first_stage_step_frac']
        self.switch_step = int(self.first_stage_step_frac * hparams['steps'])
        self.cur_model = self.stage1_model

        self.stage2_model = None    # implement in child classes

    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch

        if step < self.switch_step:
            self.cur_model = self.stage1_model
            self.cur_model.train()
            loss = self.stage1_model._compute_loss(all_i, all_x, all_y, all_a, step)
        else:
            self.cur_model = self.stage2_model
            self.cur_model.train()
            self.stage1_model.eval()
            loss = self.stage2_model._compute_loss(all_i, all_x, all_y, all_a, step, self.stage1_model)
        
        self.cur_model.optimizer.zero_grad()
        loss.backward()
        if self.cur_model.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.cur_model.network.parameters(), 1.0)
        self.cur_model.optimizer.step()

        if self.cur_model.lr_scheduler is not None:
            self.cur_model.lr_scheduler.step()

        if self.data_type == "text":
            self.cur_model.network.zero_grad()

        return {'loss': loss.item()}

    def return_feats(self, x):
        return self.cur_model.featurizer(x)
    
    def predict(self, x):
        return self.cur_model.network(x)


class JTT_Stage2(ERM): 
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super().__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

    def _compute_loss(self, i, x, y, a, step, stage1_model):
        with torch.no_grad():
            predictions = stage1_model.predict(x)

        if predictions.squeeze().ndim == 1:
            wrong_predictions = (predictions > 0).detach().ne(y).float()
        else:
            wrong_predictions = predictions.argmax(1).detach().ne(y).float()

        weights = torch.ones(wrong_predictions.shape).to(x.device).float()
        weights[wrong_predictions == 1] = self.hparams["jtt_lambda"]

        return (self.loss(self.predict(x), y) * weights).mean()


class JTT(AbstractTwoStage):
    """
    Just-train-twice (JTT) [https://arxiv.org/pdf/2107.09044.pdf]
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super().__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        self.stage2_model = JTT_Stage2(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)


class LfF(Algorithm):
    """
    Learning from Failure (LfF) [https://arxiv.org/pdf/2007.02561.pdf]
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super().__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

        self.pred_model = ERM(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None)        

        self.biased_featurizer = Featurizer(data_type, input_shape, self.hparams)
        self.biased_classifier = Classifier(
            self.biased_featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier']
        )
        self.biased_network = nn.Sequential(self.biased_featurizer, self.biased_classifier)
        self.q = self.hparams['LfF_q']
        self._init_model()

    def _init_model(self):
        self.pred_model._init_model()

        self.clip_grad = (self.data_type == "text" and self.hparams["optimizer"] == "adamw")

        if self.data_type in ["images", "tabular"]:
            self.optimizer_b = get_optimizers['sgd'](
                self.biased_network,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = None
        elif self.data_type == "text":
            self.biased_network.zero_grad()
            self.optimizer_b = get_optimizers[self.hparams["optimizer"]](
                self.biased_network,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer_b,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")

    # implemented from equation
    def GCE(self, logits, targets):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        loss = (1 - Yg.squeeze()**self.q) / self.q
        return loss

    # copied from the authors' repo
    def GCE2(self, logits, targets):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        loss = F.cross_entropy(logits, targets, reduction='none') * (Yg.squeeze().detach()**self.q)*self.q
        return loss

    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch    
        pred_logits = self.pred_model.predict(all_x) 
        biased_logits = self.biased_network(all_x)
        loss_gce = self.GCE2(biased_logits, all_y)
        ce_b = F.cross_entropy(biased_logits, all_y, reduction='none')
        ce_d = F.cross_entropy(pred_logits, all_y, reduction='none')
        weights = (ce_b/(ce_b + ce_d + 1e-8)).detach()

        self.optimizer_b.zero_grad()
        self.pred_model.optimizer.zero_grad()

        loss_pred = (ce_d * weights).mean()
        loss = loss_pred.mean() + loss_gce.mean()
        loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.biased_network.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), 1.0)
        self.optimizer_b.step()
        self.pred_model.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.pred_model.lr_scheduler.step()

        if self.data_type == "text":
            self.biased_network.zero_grad()
            self.pred_model.zero_grad()

        return {'loss': loss.item(), 'loss_pred': loss_pred.mean().item(), 'loss_gce': loss_gce.mean().item()}

    def return_feats(self, x):
        return self.pred_model.featurizer(x)

    def predict(self, x):
        return self.pred_model.predict(x)


class LISA(ERM):
    """
    Improving Out-of-Distribution Robustness via Selective Augmentation [https://arxiv.org/pdf/2201.00299.pdf]
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super().__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

    def _to_ohe(self, y):
        return F.one_hot(y, num_classes=self.num_classes)

    def _lisa_mixup_data(self, s, a, x, y, alpha):
        if (not self.data_type == "images") or self.hparams['LISA_mixup_method'] == 'mixup':
            fn = self._mix_up
        elif self.hparams['LISA_mixup_method'] == 'cutmix':
            fn = self._cut_mix_up

        all_mix_x, all_mix_y = [], []
        bs = len(x)
        # repeat until enough samples
        while sum(list(map(len, all_mix_x))) < bs:
            start_len = sum(list(map(len, all_mix_x)))
            # same label, mixup between attributes
            if s:
                # can't do intra-label mixup with only one attribute
                if len(torch.unique(a)) < 2:
                    return x, y

                for y_i in range(self.num_classes):
                    mask = y[:, y_i].squeeze().bool()
                    x_i, y_i, a_i = x[mask], y[mask], a[mask]
                    unique_a_is = torch.unique(a_i)
                    if len(unique_a_is) < 2:
                        continue

                    # if there are multiple attributes, choose a random pair
                    a_i1, a_i2 = unique_a_is[torch.randperm(len(unique_a_is))][:2]
                    mask2_1 = a_i == a_i1
                    mask2_2 = a_i == a_i2
                    all_mix_x_i, all_mix_y_i = fn(alpha, x_i[mask2_1], x_i[mask2_2], y_i[mask2_1], y_i[mask2_2])
                    all_mix_x.append(all_mix_x_i)
                    all_mix_y.append(all_mix_y_i)

            # same attribute, mixup between labels
            else:
                # can't do intra-attribute mixup with only one label
                if len(y.sum(axis=0).nonzero()) < 2:
                    return x, y

                for a_i in torch.unique(a):
                    mask = a == a_i
                    x_i, y_i = x[mask], y[mask]
                    unique_y_is = y_i.sum(axis=0).nonzero()
                    if len(unique_y_is) < 2:
                        continue

                    # if there are multiple labels, choose a random pair
                    y_i1, y_i2 = unique_y_is[torch.randperm(len(unique_y_is))][:2] 
                    mask2_1 = y_i[:, y_i1].squeeze().bool()
                    mask2_2 = y_i[:, y_i2].squeeze().bool()
                    all_mix_x_i, all_mix_y_i = fn(alpha, x_i[mask2_1], x_i[mask2_2], y_i[mask2_1], y_i[mask2_2])
                    all_mix_x.append(all_mix_x_i)
                    all_mix_y.append(all_mix_y_i)

            end_len = sum(list(map(len, all_mix_x)))
            # each attribute only has one unique label
            if end_len == start_len:
                return x, y

        all_mix_x = torch.cat(all_mix_x, dim=0)
        all_mix_y = torch.cat(all_mix_y, dim=0)

        shuffle_idx = torch.randperm(len(all_mix_x))
        return all_mix_x[shuffle_idx][:bs], all_mix_y[shuffle_idx][:bs]

    @staticmethod
    def _rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        # `np.int` was a pure alias for the builtin `int` and was removed in
        # numpy 1.24, so this is a rename, not a semantic change.
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    @staticmethod
    def _mix_up(alpha, x1, x2, y1, y2):
        # y1, y2 should be one-hot label, which means the shape of y1 and y2 should be [bsz, n_classes]
        length = min(len(x1), len(x2))
        x1 = x1[:length]
        x2 = x2[:length]
        y1 = y1[:length]
        y2 = y2[:length]

        n_classes = y1.shape[1]
        bsz = len(x1)
        l = np.random.beta(alpha, alpha, [bsz, 1])
        if len(x1.shape) == 4:
            l_x = np.tile(l[..., None, None], (1, *x1.shape[1:]))
        else:
            l_x = np.tile(l, (1, *x1.shape[1:]))
        l_y = np.tile(l, [1, n_classes])

        # mixed_input = l * x + (1 - l) * x2
        mixed_x = torch.tensor(l_x, dtype=torch.float32).to(x1.device) * x1 + torch.tensor(1-l_x, dtype=torch.float32).to(x2.device) * x2
        mixed_y = torch.tensor(l_y, dtype=torch.float32).to(y1.device) * y1 + torch.tensor(1-l_y, dtype=torch.float32).to(y2.device) * y2

        return mixed_x, mixed_y

    def _cut_mix_up(self, alpha, x1, x2, y1, y2):
        length = min(len(x1), len(x2))
        x1 = x1[:length]
        x2 = x2[:length]
        y1 = y1[:length]
        y2 = y2[:length]

        input = torch.cat([x1, x2])
        target = torch.cat([y1, y2])

        rand_index = torch.cat([torch.arange(len(y2)) + len(y1), torch.arange(len(y1))])

        lam = np.random.beta(alpha, alpha)
        target_a = target
        target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

        return input, lam * target_a + (1-lam) * target_b

    def _compute_loss(self, i, x, y, a, step):
        s = np.random.random() <= self.hparams['LISA_p_sel']
        y_ohe = self._to_ohe(y)
        if self.data_type == "text":
            feats = self.featurizer(x)
            mixed_feats, mixed_y = self._lisa_mixup_data(s, a, feats, y_ohe, self.hparams["LISA_alpha"])
            predictions = self.classifier(mixed_feats)
        else:
            mixed_x, mixed_y = self._lisa_mixup_data(s, a, x, y_ohe, self.hparams["LISA_alpha"])
            predictions = self.predict(mixed_x)

        mixed_y_float = mixed_y.type(torch.FloatTensor)
        loss_value = F.cross_entropy(predictions, mixed_y_float.to(predictions.device))
        return loss_value


# ==============================================================================
# CASE CONFIGURATION AND TRAINING LOOP
# Inlined verbatim from subpopbench/(this case's own train.py)
# ==============================================================================

# ── case configuration ───────────────────────────────────────────────────────
DATASET = "Living17"

# ERM is the benchmark's plain baseline: no subpopulation-shift mitigation at
# all. That is deliberate — it is the starting point the pipeline is supposed
# to improve on. Any of SubpopBench's ~20 algorithms is selectable here
# (see the ALGORITHMS list in this file), but two-stage methods
# (DFR / CRT / JTT) additionally need a stage-1 checkpoint and will not run
# out of the box.
ALGORITHM = "ERM"

# "no" = group attributes are HIDDEN during training (the `a` column is zeroed
# for the training split only; validation and test keep their true attributes
# so the probe and the external scorer can still measure per-group behaviour).
#
# This choice decides which published column your threshold must come from:
#   TRAIN_ATTR = "no"   -> compare against the paper's attribute-UNKNOWN
#                          training rows (`--train_attr no`)
#   TRAIN_ATTR = "yes"  -> compare against the attribute-KNOWN rows
# Mixing the two makes a good result look like a failure, or vice versa.
TRAIN_ATTR = "no"

SEED = 0

# Total optimisation steps. None = the dataset's own SubpopBench default
# (Waterbirds 5001, CelebA 30001, ...), which is what the paper's numbers were
# produced with. Override only if you accept losing comparability.
TOTAL_STEPS = None

# Hard cap on epochs. None = no cap (paper-comparable). Set an integer to
# shorten a case for faster pipeline iteration.
MAX_EPOCHS = None

# Shared data root for every case — the 16 case folders duplicate the source
# code, not the datasets. Populate it with
#   python -m subpopbench.scripts.download --data_path $SUBPOP_DATA_DIR --download
SUBPOP_DATA_DIR = os.environ.get("SUBPOP_DATA_DIR", "/mnt/workspace/data")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")

IMAGE_ARCH = "resnet_sup_in1k"
TEXT_ARCH = "bert-base-uncased"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy("file_system")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{DATASET}] algorithm={ALGORITHM} train_attr={TRAIN_ATTR} device={device}")

    hparams = default_hparams(ALGORITHM, DATASET)
    hparams.update({"image_arch": IMAGE_ARCH, "text_arch": TEXT_ARCH})
    if False:
        # SubpopBench's argparse defaults. (Upstream train.py sets
        # cmnist_label_prob from cmnist_attr_prob; the two defaults are equal,
        # so setting them explicitly here is equivalent and less confusing.)
        hparams.update({
            "cmnist_label_prob": 0.5,
            "cmnist_attr_prob": 0.5,
            "cmnist_spur_prob": 0.2,
            "cmnist_flip_prob": 0.25,
        })

    dataset_class = datasets.get_dataset_class(DATASET)
    train_dataset = dataset_class(SUBPOP_DATA_DIR, "tr", hparams, train_attr=TRAIN_ATTR)
    eval_splits = ["va"] + list(dataset_class.EVAL_SPLITS)
    eval_datasets = {s: dataset_class(SUBPOP_DATA_DIR, s, hparams) for s in eval_splits}

    total_steps = TOTAL_STEPS or dataset_class.N_STEPS
    hparams.update({"steps": total_steps})

    batch_size = hparams["batch_size"]
    steps_per_epoch = max(1, len(train_dataset) // batch_size)
    n_epochs = max(1, math.ceil(total_steps / steps_per_epoch))
    if MAX_EPOCHS is not None:
        n_epochs = min(n_epochs, MAX_EPOCHS)

    print(f"  train={len(train_dataset)} " + " ".join(
        f"{s}={len(d)}" for s, d in eval_datasets.items()))
    print(f"  batch_size={batch_size} steps/epoch={steps_per_epoch} epochs={n_epochs}")

    train_weights = None
    if hparams["group_balanced"]:
        # With TRAIN_ATTR="no" the groups degenerate to classes, matching
        # SubpopBench's behaviour.
        train_weights = np.asarray(train_dataset.weights_g, dtype=np.float64)
        train_weights /= train_weights.sum()

    train_loader = InfiniteDataLoader(
        dataset=train_dataset, weights=train_weights,
        batch_size=batch_size, num_workers=train_dataset.N_WORKERS,
    )
    eval_loaders = {
        s: FastDataLoader(dataset=d, batch_size=max(128, batch_size * 2),
                          num_workers=train_dataset.N_WORKERS)
        for s, d in eval_datasets.items()
    }

    algorithm = get_algorithm_class(ALGORITHM)(
        train_dataset.data_type, train_dataset.INPUT_SHAPE,
        train_dataset.num_labels, train_dataset.num_attributes,
        len(train_dataset), hparams, grp_sizes=train_dataset.group_sizes,
    ).to(device)

    train_iter = iter(train_loader)
    step = 0
    history = []

    for epoch in range(1, n_epochs + 1):
        algorithm.train()
        losses = []
        for _ in range(steps_per_epoch):
            i, x, y, a = next(train_iter)
            step_vals = algorithm.update((i, x.to(device), y.to(device), a.to(device)), step)
            step += 1
            if isinstance(step_vals, dict) and "loss" in step_vals:
                losses.append(float(step_vals["loss"]))

        val = eval_helper.eval_metrics(algorithm, eval_loaders["va"], device)

        # ANCHOR: original train metric — the model's own primary eval metric.
        val_accuracy = val["overall"]["accuracy"]
        # ANCHOR: original train metric — the model's own loss.
        val_loss = val["overall"]["BCE"]

        # Subpopulation-shift view of the same epoch. `worst_group_accuracy` is
        # min over the (y, a) groups; `adjusted_accuracy` is the unweighted
        # mean over those same groups (same construct, far lower variance on
        # small minority groups).
        worst_group_accuracy = val["min_group"]["accuracy"]
        adjusted_accuracy = val["adjusted_accuracy"]

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "worst_group_accuracy": worst_group_accuracy,
            "adjusted_accuracy": adjusted_accuracy,
        }
        history.append(row)
        loss_str = "n/a" if row["train_loss"] is None else f"{row['train_loss']:.4f}"
        print(
            f"  epoch {epoch:>4}/{n_epochs}  loss={loss_str}  "
            f"val_acc={val_accuracy:.4f}  worst_group={worst_group_accuracy:.4f}  "
            f"adjusted={adjusted_accuracy:.4f}",
            flush=True,
        )
        with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    algorithm.eval()
    final = {s: eval_helper.eval_metrics(algorithm, loader, device)
             for s, loader in eval_loaders.items()}
    with open(os.path.join(OUTPUT_DIR, "final_results.json"), "w") as f:
        json.dump(final, f, indent=2, default=float)

    print("\nFinal (held-out splits):")
    for s, m in final.items():
        print(f"  [{s}] avg={m['overall']['accuracy']:.4f} "
              f"worst_group={m['min_group']['accuracy']:.4f} "
              f"adjusted={m['adjusted_accuracy']:.4f}")


if __name__ == "__main__":
    sys.exit(main())

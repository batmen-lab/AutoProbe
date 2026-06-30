from dataclasses import dataclass, field
import os, os.path as osp
from typing import Any, ClassVar, Dict, List, Optional

import simple_parsing
from simple_parsing.helpers import Serializable, choice, dict_field, list_field


@dataclass
class Hparams:
    """Hyperparameters of for the run"""

    # wandb parameters
    wandb_project : str  = "classif_celeba"
    wandb_entity  : str  = "attributes_classification_celeba"       # name of the project
    save_dir      : str  = osp.join(os.getcwd())                    # directory to save wandb outputs
    weights_path  : str  = osp.join(os.getcwd(), "weights")

    # train or predict
    train : bool = True       # actually fine-tune (toy runs had this False -> ~0 F1 -> degenerate metrics)
    predict: bool = True

    gpu: int = 0
    fast_dev_run: bool = False
    # Realistic-but-small fine-tune: ~2,000 train imgs/epoch for 5 epochs.
    # Enough for the model to reach a non-trivial macro-F1 so probe metrics
    # (e.g. corruption F1-drop) are sensible rather than degenerate, without
    # the >1-day cost of the full set. EarlyStopping patience=10 > 5 epochs so
    # it always runs all 5.
    max_epochs: int = 5
    limit_train_batches: int = 32   # ~2,048 imgs/epoch (batch 64)
    limit_val_batches: int = 32     # ~2,048 val imgs for a stable metric
    val_check_interval: float = 1.0    # validate once per epoch -> one probe point/epoch

@dataclass
class TrainParams:
    """Parameters to use for the model"""
    model_name        : str         = "vit_small_patch16_224"
    pretrained        : bool        = True
    n_classes         : int         = 40
    lr : float = 5e-5   # 1e-5 barely trained (0.36 F1/30ep); 5e-5 ~doubled it -> sensible fine-tune

@dataclass
class DatasetParams:
    """Parameters to use for the model"""
    # datamodule
    num_workers       : int         = 8         # parallel data loading for the full dataset
    # root_dataset      : Optional[str] = osp.join(os.getcwd(), "assets")   # '/kaggle/working'
    root_dataset      : Optional[str] = "/home/xuanhe_linux_001/probe_cnn/CelebFaces_Attributes_Classification/assets/inputs"   # point at existing CelebA data in-place (no copy)
    batch_size        : int         = 64        # larger batch for full-scale GPU training
    input_size        : tuple       = (224, 224)   # image_size

@dataclass
class CallBackParams:
    """Parameters to use for the logging callbacks"""

    nb_image : int = 8
    early_stopping_params     : Dict[str, Any] = dict_field(
        dict(
            monitor="val/F1", 
            patience=10,
            mode="max",
            verbose=True
        )
    )
    model_checkpoint_params    : Dict[str, Any] = dict_field(
        dict(
            monitor="val/F1", 
            dirpath= osp.join(os.getcwd(), "weights"), #'/kaggle/working/', 
            filename="best-model",
            mode="max",
            verbose=True
        )
    )

@dataclass
class InferenceParams:
    """Parameters to use for the inference"""
    model_name        : str         = "vit_small_patch16_224"
    pretrained        : bool        = True
    n_classes         : int         = 40 
    ckpt_path: Optional[str] = osp.join(os.getcwd(), "weights", "ViTsmall.ckpt") 
    output_root :  str = osp.join(os.getcwd(), "output")

@dataclass
class SVMParams:
    """Parameters to edit for SVM training"""
    json_file           : str       = "outputs_stylegan/stylegan3/scores_stylegan3.json"
    np_file             : str       = "outputs_stylegan/stylegan3/z.npy"
    output_dir          : str       = "trained_boundaries_z_sg3"
    latent_space_dim    : int       = 512
    equilibrate         : bool      = False

@dataclass
class Parameters:
    """base options."""

    hparams       : Hparams         = field(default_factory=Hparams)
    data_param    : DatasetParams   = field(default_factory=DatasetParams)
    callback_param: CallBackParams  = field(default_factory=CallBackParams)
    train_param   : TrainParams     = field(default_factory=TrainParams)
    inference_param : InferenceParams = field(default_factory=InferenceParams)
    svm_params      : SVMParams = field(default_factory=SVMParams)

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance

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
    train : bool = False
    predict: bool = True

    gpu: int = 0
    fast_dev_run: bool = False
    # Lightweight fine-tune: we start from the existing ViTsmall checkpoint, so a
    # short pass is enough. One training run sees ~400 images
    # (limit_train_batches * batch_size = 50 * 8) for a single epoch.
    max_epochs: int = 5
    limit_train_batches: int = 20      # number of train batches per epoch (int = batch count)
    limit_val_batches: int = 10        # cap validation so it stays fast
    val_check_interval: float = 1.0    # validate once at the end of the (short) epoch

@dataclass
class TrainParams:
    """Parameters to use for the model"""
    model_name        : str         = "vit_small_patch16_224"
    pretrained        : bool        = True
    n_classes         : int         = 40 
    lr : int = 0.00001

@dataclass
class DatasetParams:
    """Parameters to use for the model"""
    # datamodule
    num_workers       : int         = 2         # number of workers for dataloadersint
    # root_dataset      : Optional[str] = osp.join(os.getcwd(), "assets")   # '/kaggle/working'
    root_dataset      : Optional[str] = osp.join(os.getcwd(), "assets", "inputs")   # '/kaggle/working'
    batch_size        : int         = 32        # batch_size (small fine-tune batch)
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

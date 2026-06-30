import sys; sys.path.insert(0,".")
from types import SimpleNamespace
import torch, pytorch_lightning as pl
from pytorch_lightning import Trainer
from datamodules.celebadatamodule import CelebADataModule
from lightningmodules.classification import Classification
from utils.constant import ATTRIBUTES

pl.seed_everything(42, workers=True)
ROOT="/home/xuanhe_linux_001/probe_cnn/CelebFaces_Attributes_Classification/assets/inputs"
dcfg=SimpleNamespace(root_dataset=ROOT, batch_size=64, input_size=(224,224), num_workers=8)
mcfg=SimpleNamespace(lr=5e-5, model_name="vit_small_patch16_224", pretrained=True, n_classes=40)

dm=CelebADataModule(dcfg)
model=Classification.load_from_checkpoint("weights/ViTsmall_baseline.ckpt", config=mcfg, attr_dict=ATTRIBUTES)
print("[train] fine-tuning opus-fix (penalty + reweight sampler) from baseline")
trainer=Trainer(logger=False, accelerator="gpu", devices=1, max_epochs=15,
                limit_train_batches=128, limit_val_batches=32, val_check_interval=1.0,
                enable_checkpointing=False, enable_progress_bar=False, log_every_n_steps=10)
trainer.fit(model, dm)
torch.save({"state_dict": model.state_dict()}, "weights/ViTsmall_opusfixed.ckpt")
print("[train] DONE -> weights/ViTsmall_opusfixed.ckpt")

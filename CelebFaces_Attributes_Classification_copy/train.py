import json
import os.path as osp
from datetime import datetime

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, RichProgressBar)
from pytorch_lightning.loggers import WandbLogger

from datamodules.celebadatamodule import CelebADataModule
from hparams import Parameters
from lightningmodules.classification import Classification
from utils.callbacks import MetricsCallback, WandbImageCallback
from utils.constant import ATTRIBUTES
from utils.utils_functions import create_dir


def main():
    config = Parameters.parse()

    # Lightning 2.x replaced the old `gpus=` arg with accelerator/devices.
    # Use the GPU whenever one is visible so runs don't fall back to slow CPU.
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    dataset_module = CelebADataModule(config.data_param)

    if config.hparams.train:

        # Fine-tune starting point: load our existing ViTsmall checkpoint if it is
        # present, otherwise fall back to the ImageNet-pretrained timm backbone.
        start_ckpt = config.inference_param.ckpt_path
        if start_ckpt and osp.isfile(start_ckpt):
            model = Classification.load_from_checkpoint(
                start_ckpt, config=config.train_param, attr_dict=ATTRIBUTES)
            print(f"[train] fine-tuning from checkpoint: {start_ckpt}")
        else:
            model = Classification(config.train_param, ATTRIBUTES)
            print("[train] no checkpoint found; starting from ImageNet-pretrained backbone")

        wdb_config = {}
        for k, v in vars(config).items():
            for key, value in vars(v).items():
                wdb_config[f"{k}-{key}"] = value

        wandb_logger = WandbLogger(
            config=wdb_config,
            project=config.hparams.wandb_project,
            entity=config.hparams.wandb_entity,
            allow_val_change=True,
            save_dir=config.hparams.save_dir,
        )

        callbacks = [EarlyStopping(**config.callback_param.early_stopping_params),
                     MetricsCallback(config.train_param.n_classes),
                     WandbImageCallback(config.callback_param.nb_image),
                     ModelCheckpoint(**config.callback_param.model_checkpoint_params),
                     RichProgressBar(),
                     LearningRateMonitor(),
        ]

        trainer = Trainer(logger=wandb_logger,
                          accelerator=accelerator,
                          devices=devices,
                          # auto_scale_batch_size="power",
                          callbacks=callbacks,
                          log_every_n_steps=1,
                          enable_checkpointing=True,
                          fast_dev_run=config.hparams.fast_dev_run,
                          max_epochs=config.hparams.max_epochs,
                          limit_train_batches=config.hparams.limit_train_batches,
                          limit_val_batches=config.hparams.limit_val_batches,
                          val_check_interval=config.hparams.val_check_interval,
                          )

        trainer.fit(model, dataset_module)

    if config.hparams.predict:
        output_dict = {"filenames":[], "logits":[], "converted_preds":[], "preds_with_conf":[]}
        # Load weights from the checkpoint into the model directly. Passing
        # ckpt_path= to trainer.predict() in Lightning 2.x tries to restore the
        # training-loop state, which the inference checkpoint doesn't carry
        # (KeyError: 'predict_loop').
        ckpt_path = config.inference_param.ckpt_path
        if ckpt_path and osp.isfile(ckpt_path):
            model = Classification.load_from_checkpoint(
                ckpt_path, config=config.inference_param, attr_dict=dataset_module.attr_dict)
        else:
            model = Classification(config.inference_param, dataset_module.attr_dict)
        trainer = Trainer(accelerator=accelerator, devices=devices)
        predictions = trainer.predict(model, dataset_module)
        
        output_root = config.inference_param.output_root
        create_dir(output_root)
        name_output = f"output_dict-{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.json"
        output_full_path = osp.join(output_root, name_output)

        for pred_batch in predictions:
            img_names, preds, converted_preds, converted_logits = pred_batch[0], pred_batch[1], pred_batch[2], pred_batch[3]
            # {"filenames":[], "logits":[], "converted_preds":[] }
            for i, img_name in enumerate(img_names):
                output_dict['filenames'].append(img_name)
                output_dict['logits'].append(converted_logits.tolist()[i])
                output_dict['converted_preds'].append(converted_preds[i])
                preds_with_conf = {ATTRIBUTES[idx]:round(converted_logits.tolist()[i][idx], 3) for idx in np.where(preds[i]==1.0)[0]}
                output_dict['preds_with_conf'].append(preds_with_conf)
        json.dump(output_dict, open(output_full_path, 'w'))

if __name__ == "__main__":
    main()

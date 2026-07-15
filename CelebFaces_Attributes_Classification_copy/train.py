import json
import os.path as osp
from datetime import datetime

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping,
                                         ModelCheckpoint, RichProgressBar)

from datamodules.celebadatamodule import CelebADataModule
from hparams import Parameters
from lightningmodules.classification import Classification
from utils.callbacks import MetricsCallback, WandbImageCallback
from utils.constant import ATTRIBUTES
from utils.utils_functions import create_dir


# ===========================================================================
# ===  USER ANALYSIS — DO NOT MODIFY / DO NOT REMOVE / DO NOT RENAME  ========
# ===========================================================================
# `user_analyze()` below and its call in main() are HUMAN-OWNED. It is an
# independent per-round counterfactual audit that does NOT feed the probe metric.
# See the module docstring of celeb_user_analyze.py for the full contract. The
# auto-research / fix-loop agent MUST leave this function, its arguments, its
# call site, and its `.agent_probe/.user_analysis` output completely untouched.
# ---------------------------------------------------------------------------
def user_analyze(model, datamodule, max_val_images: int = 8000, n_show: int = 100) -> None:
    """USER ANALYSIS — DO NOT MODIFY. Per-round model-checkpoint snapshot.

    The counterfactual image audit is intentionally DISABLED: the counterfactual
    probe's target attribute and masked region are not fixed ahead of time, so the
    per-round image processing cannot be written in advance. Instead this simply
    snapshots THIS round's trained model checkpoint into
    `.agent_probe/.user_analysis/` (one file per round, ckpt_round_<N>.ckpt) so the
    improving models can be inspected / reused later.
    """
    import torch as _torch
    from pathlib import Path as _Path

    out_dir = _Path(__file__).resolve().parent / '.agent_probe' / '.user_analysis'
    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Each train.py run is one round: pick the next unused checkpoint index by
        # scanning the ckpt_round_<N>.ckpt files already present.
        existing = []
        for p in out_dir.glob('ckpt_round_*.ckpt'):
            try:
                existing.append(int(p.stem.split('_')[-1]))
            except ValueError:
                continue
        round_idx = (max(existing) + 1) if existing else 1
        ckpt_path = out_dir / f'ckpt_round_{round_idx}.ckpt'

        was_training = model.training
        model.eval()

        # Prefer a full Lightning checkpoint (loadable via load_from_checkpoint);
        # fall back to a plain state_dict if the trainer is not attached.
        saved_full = False
        trainer = getattr(model, 'trainer', None)
        if trainer is not None:
            try:
                trainer.save_checkpoint(str(ckpt_path))
                saved_full = True
            except Exception:
                saved_full = False
        if not saved_full:
            _torch.save(model.state_dict(), ckpt_path)

        if was_training:
            model.train()
        print(f'[user_analyze] stored round-{round_idx} checkpoint at {ckpt_path}')
    except Exception as e:
        print('[user_analyze] skipped due to:', repr(e))
# ===========================================================================
# ===  END USER ANALYSIS — DO NOT MODIFY ABOVE THIS LINE  ===================
# ===========================================================================


def main():
    config = Parameters.parse()

    # Lightning 2.x replaced the old `gpus=` arg with accelerator/devices.
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

        # Checkpoint labelling: give each run's best checkpoint a unique, readable
        # name (run timestamp + epoch) and keep only the single best one, so
        # weights/ stops filling up with unlabelled best-model-v1/v2/... versions.
        _ckpt_params = dict(config.callback_param.model_checkpoint_params)
        _ckpt_params["filename"] = f"celeb-{datetime.now().strftime('%Y%m%d-%H%M%S')}-e{{epoch:02d}}"
        _ckpt_params.setdefault("save_top_k", 1)

        callbacks = [EarlyStopping(**config.callback_param.early_stopping_params),
                     MetricsCallback(config.train_param.n_classes),
                     WandbImageCallback(config.callback_param.nb_image),
                     ModelCheckpoint(**_ckpt_params),
                     RichProgressBar(),
        ]

        trainer = Trainer(logger=False,  # wandb disabled
                          accelerator=accelerator,
                          devices=devices,
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

        # === USER ANALYSIS CALL — DO NOT MODIFY / DO NOT REMOVE / DO NOT GATE ===
        # Human-owned independent per-round counterfactual audit. Keep exactly as-is.
        user_analyze(model, dataset_module)
        # === END USER ANALYSIS CALL ===

    if config.hparams.predict:
        output_dict = {"filenames":[], "logits":[], "converted_preds":[], "preds_with_conf":[]}
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
            for i, img_name in enumerate(img_names):
                output_dict['filenames'].append(img_name)
                output_dict['logits'].append(converted_logits.tolist()[i])
                output_dict['converted_preds'].append(converted_preds[i])
                preds_with_conf = {ATTRIBUTES[idx]:round(converted_logits.tolist()[i][idx], 3) for idx in np.where(preds[i]==1.0)[0]}
                output_dict['preds_with_conf'].append(preds_with_conf)
        json.dump(output_dict, open(output_full_path, 'w'))


if __name__ == "__main__":
    main()

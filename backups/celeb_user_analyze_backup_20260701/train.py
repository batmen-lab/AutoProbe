import json
import os.path as osp
from datetime import datetime

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (Callback, EarlyStopping,
                                         ModelCheckpoint, RichProgressBar)

from datamodules.celebadatamodule import CelebADataModule
from hparams import Parameters
from lightningmodules.classification import Classification
from prober import compute_conditional_leakage, conclude, record
from utils.callbacks import MetricsCallback, WandbImageCallback
from utils.constant import ATTRIBUTES
from utils.utils_functions import create_dir


# Probe thresholds — mean normalized CMI I(pred_a; Male | y_a_true) / H(pred_a|y_a_true).
_PROBE_STANDARD_THRESHOLD = 0.05
_PROBE_ACCEPTABLE_THRESHOLD = 0.15


class LeakageProberCallback(Callback):
    """Compute the gender-conditional leakage probe once per validation epoch and
    capture the anchor values (val/F1, val/loss) train.py's own trainer produces.
    """

    def __init__(self, datamodule, standard_threshold, acceptable_threshold):
        super().__init__()
        self.datamodule = datamodule
        self.std_thr = standard_threshold
        self.acc_thr = acceptable_threshold
        self.last_f1 = None
        self.last_loss = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        try:
            val_loader = self.datamodule.val_dataloader()
            device = next(pl_module.parameters()).device
            mean_leak, per_attr = compute_conditional_leakage(
                pl_module, val_loader, device)
        except Exception as e:
            print(f"[prober] compute_conditional_leakage failed: {e!r}")
            return

        cm = trainer.callback_metrics
        f1_t = cm.get("val/F1")
        loss_t = cm.get("val/loss")
        if f1_t is not None:
            self.last_f1 = float(f1_t.item() if hasattr(f1_t, "item") else f1_t)
        if loss_t is not None:
            self.last_loss = float(loss_t.item() if hasattr(loss_t, "item") else loss_t)

        anchor = {}
        if self.last_f1 is not None:
            anchor["val/F1"] = self.last_f1
        if self.last_loss is not None:
            anchor["val/loss"] = self.last_loss

        record(
            trainer.current_epoch, mean_leak,
            anchor=anchor or None, per_attr=per_attr,
            standard_threshold=self.std_thr,
            acceptable_threshold=self.acc_thr,
        )

        if mean_leak > self.acc_thr:
            print(f"[prober] WARN epoch {trainer.current_epoch}: "
                  f"mean gender leakage {mean_leak:.4f} > acceptable {self.acc_thr}")
        else:
            print(f"[prober] epoch {trainer.current_epoch}: "
                  f"mean gender leakage = {mean_leak:.4f}")


# ===========================================================================
# ===  USER ANALYSIS — DO NOT MODIFY / DO NOT REMOVE / DO NOT RENAME  ========
# ===========================================================================
# `user_analyze()` below and its call in main() are HUMAN-OWNED. It is an
# independent per-round counterfactual audit that does NOT feed the probe metric.
# Each round (after training) it compares the CURRENT model's validation
# predictions to the ROUND-1 BASELINE model's predictions and surfaces up to 100
# images whose prediction the fix FLIPPED out of a gender-shortcut error into a
# correct read (counter-stereotype corrections), saving a JSON + contact-sheet
# montage to `.agent_probe/.user_analysis/` (the orchestrator archives these
# per round). The auto-research / fix-loop agent MUST leave this function, its
# arguments, its call site, and its output completely untouched.
# ---------------------------------------------------------------------------
def user_analyze(model, datamodule, max_val_images: int = 8000, n_show: int = 100) -> None:
    """USER ANALYSIS — DO NOT MODIFY. Per-round counterfactual-improvement audit.

    Picks up to `n_show` validation images whose prediction the current model
    flips relative to the round-1 baseline model — prioritising counter-stereotype
    corrections (baseline wrong via the gender shortcut, current right). Writes
    `.agent_probe/.user_analysis/counterfactual_corrections.{json, _contact.png}`.
    Independent of the probe metric.
    """
    import os
    import json as _json
    from pathlib import Path as _Path
    import numpy as _np
    import torch as _torch
    from torch.utils.data import DataLoader as _DL, Subset as _Subset

    SPUR_I = 20  # Male
    TARGETS = {1: 'Arched_Eyebrows', 2: 'Attractive', 18: 'Heavy_Makeup',
               24: 'No_Beard', 33: 'Wavy_Hair', 36: 'Wearing_Lipstick'}

    out_dir = _Path(__file__).resolve().parent / '.agent_probe' / '.user_analysis'
    base_dir = out_dir / '.baseline'
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        base_dir.mkdir(parents=True, exist_ok=True)

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        val_ds = getattr(datamodule, 'val', None)
        if val_ds is None:
            datamodule.setup('fit')
            val_ds = datamodule.val
        fnames = list(val_ds.filename)
        n = min(len(val_ds), max_val_images)
        loader = _DL(_Subset(val_ds, range(n)), batch_size=128, shuffle=False, num_workers=2)

        P, Y = [], []
        with _torch.no_grad():
            for x, y in loader:
                logits = model(x.to(device))
                P.append(_torch.sigmoid(logits).float().cpu().numpy())
                Y.append(_np.asarray(y))
        P = _np.concatenate(P)
        Y = _np.concatenate(Y).astype(int)
        fnames = fnames[:n]
        male = Y[:, SPUR_I]

        # gender-leakage CMI for this model (probe-relevant context; audit only)
        def _cmi(prob):
            try:
                from sklearn.metrics import mutual_info_score
            except Exception:
                return None, {}
            pred = (prob >= 0.5).astype(int)
            _, c = _np.unique(male, return_counts=True)
            pp = c / c.sum()
            Hs = float(-_np.sum(pp * _np.log(pp)))
            per = {}
            for ti, name in TARGETS.items():
                yt = Y[:, ti]; tot = 0.0
                for v in _np.unique(yt):
                    mm = yt == v
                    if mm.sum() < 2:
                        continue
                    tot += (mm.sum() / len(yt)) * mutual_info_score(pred[mm, ti], male[mm])
                per[name] = round(tot / Hs if Hs > 1e-12 else 0.0, 4)
            mean = round(float(_np.mean(list(per.values()))), 4) if per else None
            return mean, per
        cmi_cur, cmi_per = _cmi(P)

        bpath = base_dir / 'baseline_preds.npz'
        if not bpath.exists():
            _np.savez(bpath, P=P, Y=Y, fnames=_np.array(fnames, dtype=object))
            (out_dir / 'counterfactual_corrections.json').write_text(_json.dumps({
                'stage': 'baseline (round 1)',
                'note': 'round-1 baseline predictions stored; corrections are measured from round 2 onward',
                'n_val': int(n), 'cmi_current': cmi_cur, 'cmi_per_target': cmi_per, 'n_shown': 0,
            }, indent=2))
            if was_training:
                model.train()
            print('[user_analyze] baseline predictions stored (round 1)')
            return

        d = _np.load(bpath, allow_pickle=True)
        Pb = d['P']
        m = min(len(Pb), len(P))
        Pb, Pc, Yc, fn, ml = Pb[:m], P[:m], Y[:m], fnames[:m], male[:m]

        counter_stereo, other_corr, other_flip = [], [], []
        for ti, name in TARGETS.items():
            yt = Yc[:, ti]
            bp = (Pb[:, ti] >= 0.5).astype(int)
            cp = (Pc[:, ti] >= 0.5).astype(int)
            for i in range(m):
                if bp[i] == cp[i]:
                    continue
                rec = dict(fname=str(fn[i]), attr=name, male=int(ml[i]), true=int(yt[i]),
                           p_base=round(float(Pb[i, ti]), 3), p_cur=round(float(Pc[i, ti]), 3),
                           improve=round(float(abs(Pc[i, ti] - Pb[i, ti])), 3))
                base_wrong = bp[i] != yt[i]
                cur_right = cp[i] == yt[i]
                stereo = (ml[i] == 1 and yt[i] == 1) or (ml[i] == 0 and yt[i] == 0)
                if base_wrong and cur_right and stereo:
                    counter_stereo.append(rec)
                elif base_wrong and cur_right:
                    other_corr.append(rec)
                else:
                    other_flip.append(rec)
        for lst in (counter_stereo, other_corr, other_flip):
            lst.sort(key=lambda r: -r['improve'])
        picks = (counter_stereo + other_corr + other_flip)[:n_show]

        (out_dir / 'counterfactual_corrections.json').write_text(_json.dumps({
            'stage': 'vs round-1 baseline',
            'n_val': int(m), 'cmi_current': cmi_cur, 'cmi_per_target': cmi_per,
            'n_counter_stereotype': len(counter_stereo),
            'n_other_correction': len(other_corr),
            'n_other_flip': len(other_flip),
            'n_shown': len(picks), 'picks': picks,
        }, indent=2))

        # contact-sheet montage of the flipped images (baseline b -> current f)
        if picks:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from PIL import Image
            img_dir = os.path.join(str(datamodule.root), 'celeba', 'img_align_celeba')
            cols = 10
            rows = max(1, (len(picks) + cols - 1) // cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.15))
            axes = _np.array(axes).reshape(-1)
            for j, ax in enumerate(axes):
                ax.axis('off')
                if j < len(picks):
                    p = picks[j]
                    try:
                        im = Image.open(os.path.join(img_dir, p['fname'])).convert('RGB')
                    except Exception:
                        continue
                    ax.imshow(im)
                    correct = (p['p_cur'] >= 0.5) == bool(p['true'])
                    ax.set_title(f"{p['attr'][:9]}\n{'M' if p['male'] else 'F'} true={p['true']}\n"
                                 f"b={p['p_base']}->f={p['p_cur']}", fontsize=6,
                                 color='#0a7d28' if correct else '#b00020')
            fig.suptitle(f"Counterfactual improvement — {len(picks)} val images whose prediction the fix "
                         f"flipped vs the round-1 baseline (b -> f)", fontsize=10)
            fig.tight_layout()
            fig.savefig(out_dir / 'counterfactual_corrections_contact.png', dpi=95)
            plt.close(fig)

        if was_training:
            model.train()
        print(f'[user_analyze] {len(picks)} counterfactual-corrected images saved '
              f'(counter-stereotype={len(counter_stereo)})')
    except Exception as e:
        print('[user_analyze] skipped due to:', repr(e))
# ===========================================================================
# ===  END USER ANALYSIS — DO NOT MODIFY ABOVE THIS LINE  ===================
# ===========================================================================


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

        # wandb disabled — no experiment-tracker logger (was WandbLogger).
        # Lightning still tracks metrics in callback_metrics for
        # EarlyStopping/ModelCheckpoint, so nothing else needs a logger.

        # ANCHOR: original train metric - do not remove
        # val/F1 is train.py's primary eval / checkpoint-selection metric
        # (see MetricsCallback.on_validation_epoch_end + model_checkpoint_params
        # monitor='val/F1', mode='max'). Direction: higher_is_better.
        # ANCHOR: original train metric - do not remove
        # val/loss is train.py's training loss on the validation split
        # (BCEWithLogitsLoss, logged in Classification.validation_step).
        # Direction: lower_is_better.
        leakage_prober_cb = LeakageProberCallback(
            dataset_module, _PROBE_STANDARD_THRESHOLD, _PROBE_ACCEPTABLE_THRESHOLD)
        callbacks = [EarlyStopping(**config.callback_param.early_stopping_params),
                     MetricsCallback(config.train_param.n_classes),
                     WandbImageCallback(config.callback_param.nb_image),
                     ModelCheckpoint(**config.callback_param.model_checkpoint_params),
                     RichProgressBar(),
                     leakage_prober_cb,
                     # LearningRateMonitor removed — it requires a logger, and
                     # wandb logging is disabled (logger=False).
        ]

        trainer = Trainer(logger=False,  # wandb disabled
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

        # === USER ANALYSIS CALL — DO NOT MODIFY / DO NOT REMOVE / DO NOT GATE ===
        # Human-owned independent per-round counterfactual audit. Keep exactly as-is.
        user_analyze(model, dataset_module)
        # === END USER ANALYSIS CALL ===

        # Probe: finalize the gender-conditional leakage metric and pass the
        # final-epoch anchor values through so the orchestrator's utility
        # floor is enforced against train.py's own selection metric + loss.
        final_anchor = []
        if leakage_prober_cb.last_f1 is not None:
            final_anchor.append({
                "name": "val/F1",
                "value": float(leakage_prober_cb.last_f1),
                "direction": "higher_is_better",
            })
        if leakage_prober_cb.last_loss is not None:
            final_anchor.append({
                "name": "val/loss",
                "value": float(leakage_prober_cb.last_loss),
                "direction": "lower_is_better",
            })
        conclude(_PROBE_STANDARD_THRESHOLD, _PROBE_ACCEPTABLE_THRESHOLD,
                 final_anchor=final_anchor)

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

        for pred_batch in (predictions or []):
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

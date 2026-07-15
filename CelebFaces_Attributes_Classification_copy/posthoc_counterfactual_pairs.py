"""Post-hoc counterfactual-pair visualization for the CelebA gender-shortcut run.

Reproduces the human-owned counterfactual audit that the (now ckpt-only)
user_analyze() used to do online — but run AFTER the fact from the per-round
checkpoints that user_analyze() stored:

    baseline  = round-1 model  (biased, high CMI)   -> .user_analysis/ckpt_round_2.ckpt
    improved  = round-2 model  (de-biased, low CMI) -> .user_analysis/ckpt_round_3.ckpt

It finds the exact `n_show` validation images the improvement corrected on the 6
gender-correlated target attributes (baseline predicts the attribute wrong via
the Male shortcut -> improved model predicts it right, counter-stereotype first)
and writes, for BOTH models:

    .agent_probe/.user_analysis/<role>/pair_<idx>/
        image.png          original 224x224 face
        flipped_image.png  same face, top 35% (hair/forehead) grayed = gender-cue removed
        prediction.json    that model's prob+label on image vs flipped, + a conclusion

roles = round_1_baseline (wrong via shortcut) and round_2_improved (corrected).
Nothing here feeds the probe metric.
"""

import os
import sys
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image
from torchvision import transforms as T

# Keep argparse (Parameters.parse) from consuming this script's argv.
sys.argv = [sys.argv[0]]

from hparams import Parameters
from datamodules.celebadatamodule import CelebADataModule
from lightningmodules.classification import Classification
from utils.constant import ATTRIBUTES

SPUR_I = 20  # Male
TARGETS = {1: 'Arched_Eyebrows', 2: 'Attractive', 18: 'Heavy_Makeup',
           24: 'No_Beard', 33: 'Wavy_Hair', 36: 'Wearing_Lipstick'}
MEAN, STD, SIZE = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
MASK_FRAC = 0.35  # top 35% (hair/forehead) grayed = the counterfactual gender-cue removal
N_SHOW = 100
MAX_VAL_IMAGES = 8000

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / '.agent_probe' / '.user_analysis'
BASELINE_CKPT = OUT_DIR / 'ckpt_round_2.ckpt'   # round-1 baseline (CMI ~0.129)
IMPROVED_CKPT = OUT_DIR / 'ckpt_round_3.ckpt'   # round-2 improved (CMI ~0.036)


def _load_state_dict(path, device):
    ck = torch.load(str(path), map_location=device, weights_only=False)
    return ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Parameters.parse()

    datamodule = CelebADataModule(config.data_param)
    datamodule.setup('fit')
    val_ds = datamodule.val
    fnames = list(val_ds.filename)
    n = min(len(val_ds), MAX_VAL_IMAGES)
    loader = DataLoader(Subset(val_ds, range(n)), batch_size=128, shuffle=False, num_workers=2)

    model = Classification(config.train_param, ATTRIBUTES).to(device).eval()

    def _run(sd):
        model.load_state_dict(sd)
        model.eval()
        out = []
        with torch.no_grad():
            for x, _y in loader:
                out.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy())
        return np.concatenate(out)

    # true labels (fixed across models)
    Y = []
    with torch.no_grad():
        for _x, y in loader:
            Y.append(np.asarray(y))
    Y = np.concatenate(Y).astype(int)
    fnames = fnames[:n]
    male = Y[:, SPUR_I]

    baseline_sd = _load_state_dict(BASELINE_CKPT, device)
    improved_sd = _load_state_dict(IMPROVED_CKPT, device)
    print('[posthoc] running baseline (round-1) model ...')
    Pb = _run(baseline_sd)
    print('[posthoc] running improved (round-2) model ...')
    Pc = _run(improved_sd)

    m = min(len(Pb), len(Pc))
    Pb, Pc, Yc, fn, ml = Pb[:m], Pc[:m], Y[:m], fnames[:m], male[:m]

    # gender-leakage CMI of the improved model (audit context; not the probe metric)
    def _cmi(prob):
        try:
            from sklearn.metrics import mutual_info_score
        except Exception:
            return None, {}
        pred = (prob >= 0.5).astype(int)
        _, c = np.unique(ml, return_counts=True)
        pp = c / c.sum()
        Hs = float(-np.sum(pp * np.log(pp)))
        per = {}
        for ti, name in TARGETS.items():
            yt = Yc[:, ti]; tot = 0.0
            for v in np.unique(yt):
                mm = yt == v
                if mm.sum() < 2:
                    continue
                tot += (mm.sum() / len(yt)) * mutual_info_score(pred[mm, ti], ml[mm])
            per[name] = round(tot / Hs if Hs > 1e-12 else 0.0, 4)
        mean = round(float(np.mean(list(per.values()))), 4) if per else None
        return mean, per
    cmi_cur, cmi_per = _cmi(Pc)

    # exact N_SHOW corrected images: baseline wrong -> improved right (counter-stereotype first)
    cs, oc, of = [], [], []
    for ti, name in TARGETS.items():
        yt = Yc[:, ti]
        bp = (Pb[:, ti] >= 0.5).astype(int)
        cp = (Pc[:, ti] >= 0.5).astype(int)
        for i in range(m):
            if bp[i] == cp[i]:
                continue
            rec = dict(idx_val=int(i), fname=str(fn[i]), attr=name, attr_index=int(ti),
                       male=int(ml[i]), true=int(yt[i]),
                       p_base=round(float(Pb[i, ti]), 3), p_cur=round(float(Pc[i, ti]), 3),
                       improve=round(float(abs(Pc[i, ti] - Pb[i, ti])), 3))
            bw = bp[i] != yt[i]; cr = cp[i] == yt[i]
            st = (ml[i] == 1 and yt[i] == 1) or (ml[i] == 0 and yt[i] == 0)
            (cs if (bw and cr and st) else oc if (bw and cr) else of).append(rec)
    for L in (cs, oc, of):
        L.sort(key=lambda r: -r['improve'])
    picks = (cs + oc + of)[:N_SHOW]

    (OUT_DIR / 'counterfactual_corrections.json').write_text(json.dumps({
        'stage': 'post-hoc: round-2 improved vs round-1 baseline (from stored ckpts)',
        'baseline_ckpt': BASELINE_CKPT.name, 'improved_ckpt': IMPROVED_CKPT.name,
        'n_val': int(m), 'cmi_current': cmi_cur, 'cmi_per_target': cmi_per,
        'n_counter_stereotype': len(cs), 'n_other_correction': len(oc), 'n_other_flip': len(of),
        'n_shown': len(picks), 'picks': picks,
    }, indent=2))

    # ---- render the pair folders for the baseline and the improved model ----
    norm = T.Normalize(MEAN, STD); to_t = T.ToTensor()
    img_dir = os.path.join(str(datamodule.root), 'celeba', 'img_align_celeba')
    mask_rows = int(SIZE * MASK_FRAC)

    prepared = []  # (pick, orig_pil_or_None, masked_pil_or_None)
    for p in picks:
        try:
            pil = Image.open(os.path.join(img_dir, p['fname'])).convert('RGB').resize((SIZE, SIZE))
        except Exception:
            prepared.append((p, None, None)); continue
        arr = np.array(pil).copy(); arr[:mask_rows, :, :] = 128
        prepared.append((p, pil, Image.fromarray(arr)))

    def _batch_probs(pils, attr_idxs):
        probs = np.full(len(pils), np.nan)
        valid = [(k, pl) for k, pl in enumerate(pils) if pl is not None]
        for s in range(0, len(valid), 64):
            chunk = valid[s:s + 64]
            bt = torch.stack([norm(to_t(pl)) for _k, pl in chunk]).to(device)
            with torch.no_grad():
                out = torch.sigmoid(model(bt)).float().cpu().numpy()
            for (k, _pl), row in zip(chunk, out):
                probs[k] = row[attr_idxs[k]]
        return probs

    attr_idxs = [p['attr_index'] for p, _o, _mk in prepared]

    for role, sd in (('round_1_baseline', baseline_sd), ('round_2_improved', improved_sd)):
        model.load_state_dict(sd); model.eval()
        po = _batch_probs([o for _p, o, _m in prepared], attr_idxs)
        pm = _batch_probs([mk for _p, _o, mk in prepared], attr_idxs)
        role_dir = OUT_DIR / role
        if role_dir.exists():
            shutil.rmtree(role_dir)
        role_dir.mkdir(parents=True, exist_ok=True)
        for j, (p, orig, masked) in enumerate(prepared):
            if orig is None:
                continue
            pd = role_dir / f'pair_{j:03d}'
            pd.mkdir(parents=True, exist_ok=True)
            orig.save(pd / 'image.png')
            masked.save(pd / 'flipped_image.png')
            tv = p['true']
            lo = int(po[j] >= 0.5); lm = int(pm[j] >= 0.5)
            if lo == tv:
                concl = ('CORRECT on the original face; ' +
                         ('stays correct when the hair/gender cue is masked (robust)'
                          if lm == tv else
                          'but FLIPS to wrong once the hair/gender cue is masked (still leans on the cue)'))
            else:
                concl = ('WRONG on the original face' +
                         ('; prediction changes when the hair/gender cue is masked -> relied on the gender shortcut'
                          if lm != lo else '; prediction unchanged when the cue is masked'))
            (pd / 'prediction.json').write_text(json.dumps({
                'attr': p['attr'], 'sex': 'Male' if p['male'] else 'Female', 'true_label': tv,
                'image': {'prob': round(float(po[j]), 3), 'pred': lo, 'correct': lo == tv},
                'flipped_image_hair_masked': {'prob': round(float(pm[j]), 3), 'pred': lm, 'correct': lm == tv},
                'conclusion': concl,
            }, indent=2))

    (OUT_DIR / 'summary.json').write_text(json.dumps({
        'n_pairs': len(picks), 'roles': ['round_1_baseline', 'round_2_improved'],
        'mask': f'top {int(MASK_FRAC * 100)}% (hair/forehead) grayed = counterfactual gender-cue removal',
        'note': ('same images in both roles; round_1_baseline uses the frozen round-1 model '
                 '(predictions wrong via the gender shortcut), round_2_improved uses the '
                 'de-biased round-2 model (predictions corrected). Each pair has image.png, '
                 'flipped_image.png (hair masked) and prediction.json.'),
        'counter_stereotype': len(cs), 'other_correction': len(oc), 'other_flip': len(of),
    }, indent=2))

    print(f'[posthoc] wrote {len(picks)} pair folders x2 roles '
          f'(counter-stereotype={len(cs)}, improved-model CMI={cmi_cur})')


if __name__ == '__main__':
    main()

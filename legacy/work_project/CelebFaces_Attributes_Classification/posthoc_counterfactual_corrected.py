"""Corrected counterfactual-pair selection for the CelebA gender-shortcut fix.

Proper logic (the mask removes the hair/forehead = the GENDER CUE / shortcut):
  round_1_baseline  : CORRECT on the face  -> WRONG once the cue is masked (T->F)
                      => the baseline's correct answer was propped up by the gender cue.
  round_2_improved  : CORRECT on the face  -> CORRECT with the cue masked (T->T)
                      => the de-biased model reached the same answer WITHOUT the cue.

Same image, same mask: the baseline collapses when the gender cue is removed, the
improved model does not. Pairs ranked by the baseline's flip magnitude.
"""
import os, sys, json, shutil
sys.argv = [sys.argv[0]]
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image
from torchvision import transforms as T

from legacy.work_project.CelebFaces_Attributes_Classification.hparams import Parameters
from legacy.work_project.CelebFaces_Attributes_Classification.datamodules.celebadatamodule import CelebADataModule
from legacy.work_project.CelebFaces_Attributes_Classification.lightningmodules.classification import Classification
from legacy.work_project.CelebFaces_Attributes_Classification.utils.constant import ATTRIBUTES

MALE_IDX = 20
TARGETS = {18: 'Heavy_Makeup', 36: 'Wearing_Lipstick', 2: 'Attractive',
           33: 'Wavy_Hair', 24: 'No_Beard', 1: 'Arched_Eyebrows'}
MEAN, STD, SIZE = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
MASK_FRAC = 0.35
N_SHOW = 100
MAX_IMAGES = 8000

ROOT = Path(__file__).resolve().parent
BASE_CKPT = ROOT / "weights" / "celeb-20260704-224812-eepoch=01.ckpt"   # biased baseline (CMI ~0.11)
IMP_CKPT  = ROOT / "weights" / "celeb-20260704-230002-eepoch=01.ckpt"   # de-biased improved (CMI ~0.03)
OUT = Path("/mnt/c/Users/xpan2/Desktop/autoprobe/celebF/glm_celeb/agent_probe/user_analysis")


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = Parameters.parse()
    dm = CelebADataModule(cfg.data_param); dm.setup('fit'); vd = dm.val
    fnames = list(vd.filename)
    n = min(len(vd), MAX_IMAGES)
    fnames = fnames[:n]
    loader = DataLoader(Subset(vd, range(n)), batch_size=256, shuffle=False, num_workers=2)
    Y = []
    for _x, y in loader:
        Y.append(np.asarray(y))
    Y = np.concatenate(Y).astype(int)

    def load(ck):
        m = Classification(cfg.train_param, ATTRIBUTES).to(dev).eval()
        sd = torch.load(str(ck), map_location=dev, weights_only=False)
        m.load_state_dict(sd.get('state_dict', sd)); m.eval(); return m
    base, imp = load(BASE_CKPT), load(IMP_CKPT)

    img_dir = os.path.join(str(dm.root), 'celeba', 'img_align_celeba')
    mask_rows = int(SIZE * MASK_FRAC)
    norm, to_t = T.Normalize(MEAN, STD), T.ToTensor()

    def probs(pils):
        t = torch.stack([norm(to_t(p)) for p in pils]).to(dev)
        with torch.no_grad():
            return (torch.sigmoid(base(t)).float().cpu().numpy(),
                    torch.sigmoid(imp(t)).float().cpu().numpy())

    BO, BM, IO, IM, keep = [], [], [], [], []
    bo_buf, bm_buf, idx_buf = [], [], []
    def flush():
        if not bo_buf: return
        pbo, pio = probs(bo_buf); pbm, pim = probs(bm_buf)
        BO.append(pbo); IO.append(pio); BM.append(pbm); IM.append(pim); keep.extend(idx_buf)
        bo_buf.clear(); bm_buf.clear(); idx_buf.clear()
    print(f"[posthoc] scanning {len(fnames)} val images (original + hair-masked, 2 models) ...")
    for i, fn in enumerate(fnames):
        try:
            pil = Image.open(os.path.join(img_dir, fn)).convert('RGB').resize((SIZE, SIZE))
        except Exception:
            continue
        arr = np.array(pil).copy(); arr[:mask_rows, :, :] = 128
        bo_buf.append(pil); bm_buf.append(Image.fromarray(arr)); idx_buf.append(i)
        if len(bo_buf) >= 128:
            flush()
    flush()
    BO = np.concatenate(BO); BM = np.concatenate(BM); IO = np.concatenate(IO); IM = np.concatenate(IM)
    keep = np.array(keep)
    Yk = Y[keep]; fnk = [fnames[j] for j in keep]; malek = Yk[:, MALE_IDX]

    # corrected criterion, per (image, target-attr)
    cands = []
    for k in range(len(keep)):
        for ti, name in TARGETS.items():
            yt = int(Yk[k, ti])
            bo, bm = int(BO[k, ti] >= 0.5), int(BM[k, ti] >= 0.5)
            io, im = int(IO[k, ti] >= 0.5), int(IM[k, ti] >= 0.5)
            if bo == yt and bm != yt and io == yt and im == yt:   # base T->F, improved T->T
                cands.append(dict(fname=fnk[k], attr=name, attr_index=int(ti), male=int(malek[k]), true=yt,
                                  b_img=float(BO[k, ti]), b_msk=float(BM[k, ti]),
                                  i_img=float(IO[k, ti]), i_msk=float(IM[k, ti]),
                                  flip=abs(float(BM[k, ti] - BO[k, ti]))))
    cands.sort(key=lambda r: -r['flip'])
    seen, picks = set(), []          # one (best-flip) pick per image -> unique pair_<imgid>
    for c in cands:
        if c['fname'] in seen:
            continue
        seen.add(c['fname']); picks.append(c)
        if len(picks) >= N_SHOW:
            break
    print(f"[posthoc] {len(cands)} candidate (image,attr) matches; {len(seen)} distinct images; taking {len(picks)}")

    # prepare masked/original PIL for each pick
    prepared = []
    for p in picks:
        pil = Image.open(os.path.join(img_dir, p['fname'])).convert('RGB').resize((SIZE, SIZE))
        arr = np.array(pil).copy(); arr[:mask_rows, :, :] = 128
        prepared.append((p, pil, Image.fromarray(arr)))

    def concl(po, pm, tv):
        lo, lm = int(po >= 0.5), int(pm >= 0.5)
        if lo == tv and lm == tv:
            return "CORRECT on the original face; stays correct when the hair/gender cue is masked (robust — uses the real attribute evidence)"
        if lo == tv and lm != tv:
            return "CORRECT on the original face; but FLIPS to WRONG once the hair/gender cue is masked (its correct answer relied on the gender shortcut)"
        if lo != tv and lm == tv:
            return "WRONG on the original face; becomes correct when the cue is masked"
        return "WRONG on the original face; prediction unchanged when the cue is masked"

    for role, ik, mk in (('round_1_baseline', 'b_img', 'b_msk'), ('round_2_improved', 'i_img', 'i_msk')):
        rd = OUT / role
        if rd.exists():
            shutil.rmtree(rd)
        rd.mkdir(parents=True, exist_ok=True)
        for j, (p, orig, masked) in enumerate(prepared):
            imgid = os.path.splitext(p['fname'])[0]   # CelebA image index, e.g. 012345
            pd = rd / f"pair_{imgid}"; pd.mkdir(parents=True, exist_ok=True)
            orig.save(pd / "image.png"); masked.save(pd / "flipped_image.png")
            po, pm, tv = p[ik], p[mk], p['true']
            (pd / "prediction.json").write_text(json.dumps({
                "attr": p['attr'], "sex": "Male" if p['male'] else "Female", "true_label": tv,
                "image": {"prob": round(float(po), 3), "pred": int(po >= 0.5), "correct": int(po >= 0.5) == tv},
                "flipped_image_hair_masked": {"prob": round(float(pm), 3), "pred": int(pm >= 0.5), "correct": int(pm >= 0.5) == tv},
                "conclusion": concl(po, pm, tv),
            }, indent=2))

    (OUT / "summary.json").write_text(json.dumps({
        "n_pairs": len(picks), "roles": ["round_1_baseline", "round_2_improved"],
        "selection": ("counterfactual gender-shortcut test: the top 35% (hair/forehead) is grayed = the gender "
                      "cue is removed. Pairs are chosen so the BASELINE is correct on the face but FLIPS to wrong "
                      "once the cue is masked (its correct answer leaned on the gender shortcut), while the "
                      "IMPROVED model stays correct with the cue masked (it uses the real attribute evidence). "
                      "Same image + same mask in both roles; ranked by the baseline's flip magnitude."),
        "mask": f"top {int(MASK_FRAC*100)}% (hair/forehead) grayed = gender-cue removal",
        "baseline_ckpt": BASE_CKPT.name, "improved_ckpt": IMP_CKPT.name,
        "n_candidates_total": len(cands),
    }, indent=2))
    print(f"[posthoc] wrote {len(picks)} pairs x2 roles to {OUT}")


if __name__ == "__main__":
    main()

"""Reusable `user_analyze()` for the CelebA counterfactual (gender-shortcut) probe.

This is the human-owned, protected per-round audit. Drop the function below
(between the banner comments) into a repo's train.py and call it once after
training each round:  `user_analyze(model, datamodule)`.

Behaviour
---------
Round 1 (no baseline yet): save the baseline model weights + validation
predictions, then return.

Every later (improved) round: fix the EXACT `n_show` validation images the
improvement corrected (baseline predicts the gender-correlated attribute wrong
via the Male shortcut -> current model predicts it right, counter-stereotype
first), and write one folder per image, for BOTH the frozen baseline model and
the current model:

    .agent_probe/.user_analysis/counterfactual_pairs/<role>/pair_<idx>/
        image.png          original 224x224 face
        flipped_image.png  same face with the top hair/forehead band grayed
                           (the counterfactual: remove the gender cue)
        prediction.json    that model's prob+label on image vs flipped_image
                           for the corrected attribute, plus a plain conclusion

roles = round_1_baseline (predictions wrong) and round_2_improved (corrected).
The round summary `counterfactual_corrections.json` is still written. Nothing
here feeds the probe metric.
"""


def user_analyze(model, datamodule, max_val_images: int = 8000, n_show: int = 100) -> None:
    """USER ANALYSIS — DO NOT MODIFY. Per-round counterfactual-improvement audit."""
    import os
    import json as _json
    from pathlib import Path as _Path
    import numpy as _np
    import torch as _torch
    from torch.utils.data import DataLoader as _DL, Subset as _Subset

    SPUR_I = 20  # Male
    TARGETS = {1: 'Arched_Eyebrows', 2: 'Attractive', 18: 'Heavy_Makeup',
               24: 'No_Beard', 33: 'Wavy_Hair', 36: 'Wearing_Lipstick'}
    MEAN, STD, SIZE = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
    MASK_FRAC = 0.35  # top 35% (hair/forehead) grayed = the counterfactual gender-cue removal

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

        def _run(mdl):
            out = []
            with _torch.no_grad():
                for x, _y in loader:
                    out.append(_torch.sigmoid(mdl(x.to(device))).float().cpu().numpy())
            return _np.concatenate(out)

        Y = []
        with _torch.no_grad():
            for _x, y in loader:
                Y.append(_np.asarray(y))
        Y = _np.concatenate(Y).astype(int)
        fnames = fnames[:n]
        male = Y[:, SPUR_I]
        Pc = _run(model)

        # gender-leakage CMI summary (audit context; not the probe metric)
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
        cmi_cur, cmi_per = _cmi(Pc)

        bmodel_path = base_dir / 'baseline_model.pt'
        bpred_path = base_dir / 'baseline_preds.npz'
        if not bpred_path.exists() or not bmodel_path.exists():
            _torch.save(model.state_dict(), bmodel_path)
            _np.savez(bpred_path, P=Pc, Y=Y, fnames=_np.array(fnames, dtype=object))
            (out_dir / 'counterfactual_corrections.json').write_text(_json.dumps({
                'stage': 'baseline (round 1)',
                'note': 'baseline model + predictions stored; pair folders are generated from round 2 onward',
                'n_val': int(n), 'cmi_current': cmi_cur, 'cmi_per_target': cmi_per, 'n_shown': 0,
            }, indent=2))
            if was_training:
                model.train()
            print('[user_analyze] baseline model + preds stored (round 1)')
            return

        d = _np.load(bpred_path, allow_pickle=True)
        Pb = d['P']
        m = min(len(Pb), len(Pc))
        Pb, Pc, Yc, fn, ml = Pb[:m], Pc[:m], Y[:m], fnames[:m], male[:m]

        # exact n_show corrected images: baseline wrong -> current right (counter-stereotype first)
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
        picks = (cs + oc + of)[:n_show]

        (out_dir / 'counterfactual_corrections.json').write_text(_json.dumps({
            'stage': 'vs round-1 baseline', 'n_val': int(m),
            'cmi_current': cmi_cur, 'cmi_per_target': cmi_per,
            'n_counter_stereotype': len(cs), 'n_other_correction': len(oc), 'n_other_flip': len(of),
            'n_shown': len(picks), 'picks': picks,
        }, indent=2))

        # ---- render the 100 pair folders for the baseline and the current model ----
        import shutil as _sh
        from PIL import Image
        from torchvision import transforms as _T
        norm = _T.Normalize(MEAN, STD); to_t = _T.ToTensor()
        img_dir = os.path.join(str(datamodule.root), 'celeba', 'img_align_celeba')
        mask_rows = int(SIZE * MASK_FRAC)

        # prepare each pick's original + hair-masked PIL once (model-independent)
        prepared = []  # (pick, orig_pil_or_None, masked_pil_or_None)
        for p in picks:
            try:
                pil = Image.open(os.path.join(img_dir, p['fname'])).convert('RGB').resize((SIZE, SIZE))
            except Exception:
                prepared.append((p, None, None)); continue
            arr = _np.array(pil).copy(); arr[:mask_rows, :, :] = 128
            prepared.append((p, pil, Image.fromarray(arr)))

        def _batch_probs(mdl, pils, attr_idxs):
            probs = _np.full(len(pils), _np.nan)
            valid = [(k, pl) for k, pl in enumerate(pils) if pl is not None]
            for s in range(0, len(valid), 64):
                chunk = valid[s:s + 64]
                bt = _torch.stack([norm(to_t(pl)) for _k, pl in chunk]).to(device)
                with _torch.no_grad():
                    out = _torch.sigmoid(mdl(bt)).float().cpu().numpy()
                for (k, _pl), row in zip(chunk, out):
                    probs[k] = row[attr_idxs[k]]
            return probs

        attr_idxs = [p['attr_index'] for p, _o, _mk in prepared]
        cur_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        baseline_sd = _torch.load(bmodel_path, map_location=device)
        pairs_root = out_dir / 'counterfactual_pairs'

        for role, sd in (('round_1_baseline', baseline_sd), ('round_2_improved', cur_sd)):
            model.load_state_dict(sd); model.eval()
            po = _batch_probs(model, [o for _p, o, _m in prepared], attr_idxs)
            pm = _batch_probs(model, [mk for _p, _o, mk in prepared], attr_idxs)
            role_dir = pairs_root / role
            if role_dir.exists():
                _sh.rmtree(role_dir)
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
                (pd / 'prediction.json').write_text(_json.dumps({
                    'attr': p['attr'], 'sex': 'Male' if p['male'] else 'Female', 'true_label': tv,
                    'image': {'prob': round(float(po[j]), 3), 'pred': lo, 'correct': lo == tv},
                    'flipped_image_hair_masked': {'prob': round(float(pm[j]), 3), 'pred': lm, 'correct': lm == tv},
                    'conclusion': concl,
                }, indent=2))
        model.load_state_dict(cur_sd); model.eval()  # restore

        (pairs_root / 'summary.json').write_text(_json.dumps({
            'n_pairs': len(picks), 'roles': ['round_1_baseline', 'round_2_improved'],
            'mask': f'top {int(MASK_FRAC * 100)}% (hair/forehead) grayed = counterfactual gender-cue removal',
            'note': ('same images in both roles; round_1_baseline uses the frozen round-1 model '
                     '(predictions wrong via the gender shortcut), round_2_improved uses the current '
                     'de-biased model (predictions corrected). Each pair has image.png, flipped_image.png '
                     '(hair masked) and prediction.json.'),
            'counter_stereotype': len(cs), 'other_correction': len(oc), 'other_flip': len(of),
        }, indent=2))

        if was_training:
            model.train()
        print(f'[user_analyze] wrote {len(picks)} pair folders x2 roles (counter-stereotype={len(cs)})')
    except Exception as e:
        print('[user_analyze] skipped due to:', repr(e))

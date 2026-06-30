"""Explicit bias-reduction visual: per-group recall (deaths caught) baseline vs fix."""
from pathlib import Path
import numpy as np, torch
from scipy import sparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA=Path("/home/xuanhe_linux_001/aim_frontend_experiment3/aim/examples/agent_example_repos/mimic/data")
SNAP=Path("/home/xuanhe_linux_001/AutoProbe/ethnic_auroc/ckpt_snapshots")
ETH=["white","black","hispanic","asian","other"]
tfidf=sparse.load_npz(DATA/"val_tfidf.npz").tocsr()
meta=np.load(DATA/"val_meta.npz",allow_pickle=True)
eth=np.asarray(meta["eth"]); y=np.asarray(meta["labels"]).astype(int).ravel()

def scores(ckpt):
    sd=torch.load(ckpt,map_location="cpu",weights_only=True)
    W=sd["linear.weight"].numpy().astype(np.float64).ravel(); b=float(sd["linear.bias"].numpy().ravel()[0])
    X=tfidf if W.shape[0]==tfidf.shape[1] else sparse.hstack([tfidf,sparse.csr_matrix(eth.astype(np.float64))]).tocsr()
    return 1/(1+np.exp(-(X.dot(W)+b)))

def recall_counts(p):
    thr=np.quantile(p,0.80); pred=(p>=thr).astype(int)
    out={}
    for c,nm in enumerate(ETH):
        m=eth[:,c]==1; pos=m&(y==1)
        caught=int(pred[pos].sum()); total=int(pos.sum())
        out[nm]=(caught/total if total else np.nan, caught, total)
    return out, thr

base,thrb = recall_counts(scores(SNAP/"glm_round1.pt"))
fix,thrf  = recall_counts(scores(SNAP/"glm_round3.pt"))

x=np.arange(len(ETH)); w=0.38
fig,ax=plt.subplots(figsize=(11,6))
rb=[base[g][0] for g in ETH]; rf=[fix[g][0] for g in ETH]
b1=ax.bar(x-w/2, rb, w, label="Baseline (biased)", color="#c44e52")
b2=ax.bar(x+w/2, rf, w, label="After fairness fix", color="#4c72b0")
for i,g in enumerate(ETH):
    ax.annotate(f"{base[g][1]}/{base[g][2]}", (x[i]-w/2, rb[i]), ha="center", va="bottom", fontsize=8.5)
    ax.annotate(f"{fix[g][1]}/{fix[g][2]}", (x[i]+w/2, rf[i]), ha="center", va="bottom", fontsize=8.5)
# gap brackets
def gapline(vals,xpos,txt,color):
    lo,hi=min(vals),max(vals)
    ax.annotate("", (xpos,lo),(xpos,hi), arrowprops=dict(arrowstyle="<->",color=color,lw=1.5))
    ax.text(xpos+0.06,(lo+hi)/2,txt,color=color,fontsize=9,va="center")
gapline(rb, -0.46, f"gap={max(rb)-min(rb):.2f}", "#c44e52")
gapline(rf, len(ETH)-0.54, f"gap={max(rf)-min(rf):.2f}", "#4c72b0")
ax.set_xticks(x); ax.set_xticklabels(ETH)
ax.set_ylabel("Recall = fraction of actual ICU deaths flagged\n(at a shared ~20%-alert threshold)")
ax.set_title("Reducing ethnic bias in MIMIC mortality alerts\nPer-group recall, baseline vs fix  (labels = deaths caught / total deaths)")
ax.set_ylim(0,1.0); ax.legend(loc="upper right"); ax.grid(axis="y",alpha=0.3)
fig.tight_layout()
out=Path("/home/xuanhe_linux_001/AutoProbe/ethnic_auroc/glm/glm_recall_bias_before_after.png")
fig.savefig(out,dpi=140); print("wrote",out)
print("baseline thr",round(thrb,3),"fix thr",round(thrf,3))
for g in ETH: print(f"  {g:9} baseline {base[g][1]}/{base[g][2]} -> fix {fix[g][1]}/{fix[g][2]}")

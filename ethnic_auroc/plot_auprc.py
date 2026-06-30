import csv, glob, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GLM = Path("/home/xuanhe_linux_001/AutoProbe/ethnic_auroc/glm")
ETH = ["white","black","hispanic","asian","other"]
files = sorted(GLM.glob("glm_*_pergroup.csv"), key=lambda p: int(re.search(r"glm_(\d+)_",p.name).group(1)))
rounds = [int(re.search(r"glm_(\d+)_",p.name).group(1)) for p in files]
# auprc[round][group]
data = {}
for p,r in zip(files,rounds):
    rows = {row["ethnicity"]: float(row["auprc"]) for row in csv.DictReader(p.open())}
    data[r] = [rows[g] for g in ETH]

x = np.arange(len(ETH))
nR = len(rounds)
w = 0.8/nR
fig, ax = plt.subplots(figsize=(10,5.5))
labels = {1:"R1 baseline",2:"R2 drop one-hot",3:"R3 +balanced (ACCEPT)",4:"R4 -pos_weight (rev)",5:"R5 +sched (rev)"}
colors = plt.cm.viridis(np.linspace(0.1,0.85,nR))
for i,r in enumerate(rounds):
    bars = ax.bar(x + i*w - 0.4 + w/2, data[r], w, label=labels.get(r,f"R{r}"), color=colors[i])
ax.set_xticks(x); ax.set_xticklabels([f"{g}" for g in ETH])
ax.set_ylabel("AUPRC (average precision)")
ax.set_title("mimic per-ethnicity AUPRC across fix rounds (GLM, clean baseline)\nindependent yardstick — fairness fix maintained per-group AUPRC")
ax.legend(fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(max(v) for v in data.values())*1.15)
fig.tight_layout()
out = GLM / "glm_auprc_per_group.png"
fig.savefig(out, dpi=130)
print("wrote", out)

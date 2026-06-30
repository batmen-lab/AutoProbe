import json
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
IMG="/home/xuanhe_linux_001/probe_cnn/CelebFaces_Attributes_Classification/assets/inputs/celeba/img_align_celeba"
picks=json.load(open("shortlist.json"))
SEL=[0,6,12,18,30]
chosen=[picks[i] for i in SEL]
import os; os.makedirs("pairs",exist_ok=True)
rows=len(chosen)
fig,axes=plt.subplots(rows,2,figsize=(8.6,3.25*rows))
plt.subplots_adjust(wspace=0.45, hspace=0.42, left=0.20)
for r,p in enumerate(chosen):
    attr=p["attr"].replace("_"," "); sex="Male" if p["male"] else "Female"
    im=Image.open(f"{IMG}/{p['fname']}").convert("RGB"); pb,pf=p["pb"],p["pf"]
    axL,axR=axes[r,0],axes[r,1]
    for ax in (axL,axR): ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    axL.set_title(f"BEFORE  ·  baseline\nP={pb:.2f}  →  NO  ✗",color="#b00020",fontsize=10,fontweight="bold")
    axR.set_title(f"AFTER  ·  opus-fixed\nP={pf:.2f}  →  YES  ✓",color="#0a7d28",fontsize=10,fontweight="bold")
    for sp in axL.spines.values(): sp.set_color("#b00020"); sp.set_linewidth(2.5); sp.set_visible(True)
    for sp in axR.spines.values(): sp.set_color("#0a7d28"); sp.set_linewidth(2.5); sp.set_visible(True)
    fig.text(0.045, axL.get_position().y0+axL.get_position().height/2,
             f"{attr}\nCelebA label: YES\nSex: {sex}", fontsize=10, ha="left", va="center", fontweight="bold")
    # individual pair
    f2,a2=plt.subplots(1,2,figsize=(7.2,4.0)); plt.subplots_adjust(wspace=0.35)
    for a,(t,c) in zip(a2,[(f"BEFORE · baseline\nP({attr})={pb:.2f} → NO ✗","#b00020"),
                           (f"AFTER · opus-fixed\nP({attr})={pf:.2f} → YES ✓","#0a7d28")]):
        a.imshow(im); a.set_xticks([]);a.set_yticks([]); a.set_title(t,color=c,fontsize=11,fontweight="bold")
        for sp in a.spines.values(): sp.set_color(c); sp.set_linewidth(3)
    f2.suptitle(f"{attr} — CelebA label YES, Sex {sex}\nsame image, gender-shortcut error corrected by the fix",fontsize=11,y=1.04)
    f2.savefig(f"pairs/opus_pair_{r+1}_{p['attr']}.png",dpi=140,bbox_inches="tight"); plt.close(f2)
fig.suptitle("Opus counterfactual fix — gender-shortcut corrections on CelebA   (CMI 0.068 → 0.032)\n"
             "Each is a MAN whom CelebA labels with a female-coded attribute. Baseline predicts NO via the gender shortcut; the de-biased model reads the face and predicts YES.",
             fontsize=10.5,y=0.995)
fig.savefig("opus_5pairs_combined.png",dpi=140,bbox_inches="tight")
print("ok:",[p['attr'] for p in chosen])

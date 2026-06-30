import sys; sys.path.insert(0,".")
import numpy as np, json
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from utils.constant import ATTRIBUTES
NI={v:k for k,v in ATTRIBUTES.items()}
IMG="/home/xuanhe_linux_001/probe_cnn/CelebFaces_Attributes_Classification/assets/inputs/celeba/img_align_celeba"
d=np.load("preds.npz",allow_pickle=True); Pb,Pf,Y,fn=d["Pb"],d["Pf"],d["Y"],d["fnames"]
male=Y[:,NI["Male"]].astype(int)
TARGETS=["Heavy_Makeup","Wearing_Lipstick","Wavy_Hair","Arched_Eyebrows","Attractive","No_Beard"]
# build buckets: (attr, dir) dir A = man w/ female-coded true=1 ; dir B = woman w/ true=0
picks=[]
for t in TARGETS:
    ti=NI[t]; yt=Y[:,ti].astype(int)
    for dirn,(msel,tsel) in {"A_man_true1":(1,1),"B_woman_true0":(0,0)}.items():
        idx=[i for i in range(len(Y)) if male[i]==msel and yt[i]==tsel
             and (int(Pb[i,ti]>=.5)!=tsel) and (int(Pf[i,ti]>=.5)==tsel)]
        idx.sort(key=lambda i:-abs(Pf[i,ti]-Pb[i,ti]))
        for i in idx[:3]:
            picks.append(dict(i=int(i),fname=str(fn[i]),attr=t,dir=dirn,male=int(male[i]),true=int(yt[i]),
                              pb=round(float(Pb[i,ti]),2),pf=round(float(Pf[i,ti]),2)))
json.dump(picks,open("shortlist.json","w"),indent=1)
n=len(picks); cols=6; rows=(n+cols-1)//cols
fig,axes=plt.subplots(rows,cols,figsize=(cols*2.2,rows*2.6))
for k,ax in enumerate(axes.ravel()):
    ax.axis("off")
    if k<n:
        p=picks[k]; im=Image.open(f"{IMG}/{p['fname']}").convert("RGB")
        ax.imshow(im)
        ax.set_title(f"#{k} {p['attr'][:10]}\n{'M' if p['male'] else 'F'} true={p['true']}\nb={p['pb']} f={p['pf']}",fontsize=7)
fig.tight_layout(); fig.savefig("contact_sheet.png",dpi=90); print("wrote contact_sheet.png  n=",n)

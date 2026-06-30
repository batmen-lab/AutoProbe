import sys; sys.path.insert(0,".")
from types import SimpleNamespace
import numpy as np, torch, json
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import mutual_info_score
from datasets.celeba import MyCelebA
from lightningmodules.classification import Classification
from utils.constant import ATTRIBUTES

ROOT="/home/xuanhe_linux_001/probe_cnn/CelebFaces_Attributes_Classification/assets/inputs"
NI={v:k for k,v in ATTRIBUTES.items()}
SPUR="Male"; TARGETS=["Heavy_Makeup","Wearing_Lipstick","Attractive","Wavy_Hair","No_Beard","Arched_Eyebrows"]
mcfg=SimpleNamespace(lr=5e-5, model_name="vit_small_patch16_224", pretrained=True, n_classes=40)
mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean,std)])
ds=MyCelebA(ROOT, split="valid", transform=tf)
fnames=list(ds.filename)
dl=DataLoader(ds,batch_size=128,num_workers=8,shuffle=False)

def load(ck):
    sd=torch.load(ck,map_location="cpu",weights_only=False)
    if isinstance(sd,dict) and "pytorch-lightning_version" in sd:
        m=Classification.load_from_checkpoint(ck,config=mcfg,attr_dict=ATTRIBUTES)
    else:
        m=Classification(mcfg,ATTRIBUTES); m.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    m.eval().cuda(); return m
def run(m):
    P=[]; Y=[]
    with torch.no_grad():
        for x,y in dl:
            P.append(torch.sigmoid(m(x.cuda())).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)

base=load("weights/ViTsmall_baseline.ckpt"); fix=load("weights/ViTsmall_opusfixed.ckpt")
Pb,Y=run(base); Pf,_=run(fix)
np.savez("preds.npz", Pb=Pb, Pf=Pf, Y=Y, fnames=np.array(fnames))

def cmi(P,Y):
    pred=(P>=0.5).astype(int); ys=Y[:,NI[SPUR]].astype(int)
    def H(l):
        _,c=np.unique(l,return_counts=True); p=c/c.sum(); return float(-np.sum(p*np.log(p)))
    hs=H(ys); sc=[]
    for t in TARGETS:
        ti=NI[t]; pt=pred[:,ti]; yt=Y[:,ti].astype(int); tot=0.0
        for v in np.unique(yt):
            m=yt==v; w=m.sum()/len(yt)
            if m.sum()<2: continue
            tot+=w*mutual_info_score(pt[m],ys[m])
        sc.append(tot/hs if hs>1e-12 else 0)
    return float(np.mean(sc)), {t:round(s,4) for t,s in zip(TARGETS,sc)}

cb,db=cmi(Pb,Y); cf,df=cmi(Pf,Y)
print(f"CMI baseline={cb:.4f}  opus-fixed={cf:.4f}  (lower better)")
print("  per-target baseline:",db); print("  per-target fixed   :",df)

# counter-stereotype corrections: baseline wrong (gender shortcut), fixed right
male=Y[:,NI[SPUR]].astype(int)
cands=[]
for t in TARGETS:
    ti=NI[t]; yt=Y[:,ti].astype(int)
    for i in range(len(Y)):
        pb,pf=Pb[i,ti],Pf[i,ti]; tv=yt[i]
        base_pred=int(pb>=0.5); fix_pred=int(pf>=0.5)
        if base_pred!=tv and fix_pred==tv:           # fix corrects a baseline error
            # is it gender-stereotype aligned? man(1) w/ female-coded true=1, or woman(0) w/ true=0
            stereo = (male[i]==1 and tv==1) or (male[i]==0 and tv==0)
            if stereo:
                improve=abs(pf-pb)
                cands.append(dict(fname=fnames[i], attr=t, male=int(male[i]), true=int(tv),
                                  p_base=round(float(pb),3), p_fix=round(float(pf),3), improve=round(float(improve),3)))
cands.sort(key=lambda c:-c["improve"])
json.dump(cands[:60], open("candidates.json","w"), indent=1)
print(f"\n{len(cands)} counter-stereotype corrections. top 15:")
for c in cands[:15]: print(" ",c)

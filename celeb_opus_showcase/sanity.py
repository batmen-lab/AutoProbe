import sys, types, torch
sys.path.insert(0, ".")
from types import SimpleNamespace
from datasets.celeba import MyCelebA
from torchvision import transforms
from lightningmodules.classification import Classification
from utils.constant import ATTRIBUTES

ROOT="/home/xuanhe_linux_001/probe_cnn/CelebFaces_Attributes_Classification/assets/inputs"
cfg=SimpleNamespace(lr=5e-5, model_name="vit_small_patch16_224", pretrained=True, n_classes=40)
m=Classification.load_from_checkpoint("weights/ViTsmall_baseline.ckpt", config=cfg, attr_dict=ATTRIBUTES)
m.eval().cuda()
mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean,std)])
ds=MyCelebA(ROOT, split="valid", transform=tf)
print("val size:",len(ds),"| n filenames:",len(ds.filename),"| ex fname:",ds.filename[0])
xb=torch.stack([ds[i][0] for i in range(8)]).cuda()
yb=torch.stack([ds[i][1] for i in range(8)])
with torch.no_grad(): p=torch.sigmoid(m(xb)).cpu()
ni={v:k for k,v in ATTRIBUTES.items()}
print("logits shape:",p.shape)
for i in range(8):
    print(f"{ds.filename[i]} Male={int(yb[i,ni['Male']])} HM_true={int(yb[i,ni['Heavy_Makeup']])} P(HM)={p[i,ni['Heavy_Makeup']]:.3f} Lip_true={int(yb[i,ni['Wearing_Lipstick']])} P(Lip)={p[i,ni['Wearing_Lipstick']]:.3f}")

"""
Run a trained ROCOv2 model on one image (or a folder of images).

  python predict.py --model_dir rocov2_model --model fusion --image xray.jpg --caption "chest x-ray showing ..."
  python predict.py --model_dir rocov2_model --model vit --image xray.jpg --topn 8

--model_dir must contain: <model>.pt, <model>_config.json, label_vocab.json
(and tfidf.pkl for the fusion model; cui_mapping.csv is optional, gives readable concept names).
No dataset and no internet needed - pretrained weights are NOT downloaded, your trained weights are loaded.
"""
import os, json, argparse, glob
import numpy as np, torch, torch.nn as nn
import torchvision.transforms as T, torchvision.models as tvm, timm
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self, k):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 256), blk(256, 256),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, k))
    def forward(self, x, t=None): return self.head(self.f(x))

class ResNet50(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.m = tvm.resnet50(weights=None)
        self.m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, k))
    def forward(self, x, t=None): return self.m(x)

class ViT(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=k, drop_rate=0.2)
    def forward(self, x, t=None): return self.m(x)

class Fusion(nn.Module):
    def __init__(self, k, tdim):
        super().__init__()
        self.v = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        self.t = nn.Sequential(nn.Linear(tdim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(768 + 256, k))
    def forward(self, x, t): return self.head(torch.cat([self.v(x), self.t(t)], 1))

def load(model_dir, name, device):
    cfg = json.load(open(f"{model_dir}/{name}_config.json"))
    vocab = json.load(open(f"{model_dir}/label_vocab.json"))
    k = cfg["num_labels"]
    model = {"cnn": lambda: SimpleCNN(k), "resnet50": lambda: ResNet50(k), "vit": lambda: ViT(k),
             "fusion": lambda: Fusion(k, cfg["text_dim"])}[name]()
    sd = torch.load(f"{model_dir}/{name}.pt", map_location="cpu")
    sd = {a: (b.float() if b.is_floating_point() else b) for a, b in sd.items()}  # fp16 files -> fp32
    model.load_state_dict(sd)
    tfidf = None
    if name == "fusion":
        import joblib
        tfidf = joblib.load(f"{model_dir}/tfidf.pkl")
    names = {}
    mp = f"{model_dir}/cui_mapping.csv"
    if os.path.exists(mp):
        import pandas as pd
        d = pd.read_csv(mp); names = dict(zip(d.iloc[:, 0].astype(str), d.iloc[:, 1].astype(str)))
    return model.to(device).eval(), cfg, vocab, tfidf, names

TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

@torch.no_grad()
def predict_image(model, cfg, vocab, tfidf, names, path, caption="", topn=10, device="cpu"):
    x = TF(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    if tfidf is not None:
        t = torch.from_numpy(tfidf.transform([caption]).toarray().astype(np.float32)).to(device)
    else:
        t = torch.zeros(1, 1, device=device)
    p = torch.sigmoid(model(x, t))[0].cpu().numpy()
    th = cfg["threshold"]
    keep = [i for i in np.argsort(-p) if p[i] >= th] or [int(p.argmax())]   # at least one concept
    return [(vocab[i], names.get(vocab[i], ""), float(p[i])) for i in keep[:topn]]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model", default="fusion", choices=["cnn", "resnet50", "vit", "fusion"])
    ap.add_argument("--image", required=True, help="image file or folder")
    ap.add_argument("--caption", default="", help="caption text (only used by the fusion model)")
    ap.add_argument("--topn", type=int, default=10)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, vocab, tfidf, names = load(a.model_dir, a.model, dev)
    files = [a.image] if os.path.isfile(a.image) else sorted(
        f for e in ("jpg", "jpeg", "png") for f in glob.glob(os.path.join(a.image, f"*.{e}")))
    for f in files:
        print(f"\n{f}")
        for cui, name, prob in predict_image(model, cfg, vocab, tfidf, names, f, a.caption, a.topn, dev):
            print(f"  {prob:.3f}  {cui}  {name}")

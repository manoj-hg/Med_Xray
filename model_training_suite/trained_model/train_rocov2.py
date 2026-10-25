"""
Multimodal Medical Image Understanding on ROCOv2
Task: multi-label concept (CUI) detection, trained with
  1) simple CNN (from scratch)   2) ResNet50 (ImageNet pretrained)
  3) ViT-B/16 (ImageNet pretrained)   4) ViT + radiology caption text (multimodal fusion)

Usage (Colab/Kaggle GPU):
  python train_rocov2.py --data /content/rocov2 --models cnn resnet50 vit fusion --epochs 3 --out results
Saves per model in --out:  <name>.pt, <name>_history.csv, <name>_config.json
Saves shared files in --out: label_vocab.json, tfidf.pkl, comparison.csv, plots
"""
import os, glob, json, argparse, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import joblib
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T, torchvision.models as tvm, timm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", default="results")
p.add_argument("--models", nargs="+", default=["cnn", "resnet50", "vit", "fusion"])
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--topk", type=int, default=500, help="number of most frequent CUIs used as labels")
p.add_argument("--subset", type=int, default=0, help="use only N train images (quick test)")
p.add_argument("--workers", type=int, default=2)
p.add_argument("--eval_only", action="store_true", help="skip training; load saved checkpoints, make plots")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); np.random.seed(42)

# ---------------------------------------------------------------- data
def find(name):
    hits = glob.glob(os.path.join(args.data, "**", name), recursive=True)
    return hits[0] if hits else None

def index_images():
    idx = {}
    for ext in ("jpg", "jpeg", "png"):
        for f in glob.glob(os.path.join(args.data, "**", f"*.{ext}"), recursive=True):
            idx[os.path.splitext(os.path.basename(f))[0]] = f
    return idx

img_index = index_images()
print("images found:", len(img_index))

def load_split(split):
    c = find(f"{split}_concepts.csv")
    if c is None:
        return None
    df = pd.read_csv(c)
    cap = find(f"{split}_captions.csv")
    df = df.merge(pd.read_csv(cap), on="ID", how="left") if cap else df.assign(Caption="")
    df["Caption"] = df["Caption"].fillna("")
    df["CUIs"] = df["CUIs"].fillna("")
    df = df[df.ID.isin(img_index)].reset_index(drop=True)
    return df

train, valid, test = load_split("train"), load_split("valid"), load_split("test")
if train is None or test is None:
    raise SystemExit("train/test csv files not found under --data")
if args.subset:
    train = train.sample(args.subset, random_state=0).reset_index(drop=True)
if valid is None:
    valid = train.sample(frac=0.1, random_state=0)
    train = train.drop(valid.index).reset_index(drop=True); valid = valid.reset_index(drop=True)
print(f"train {len(train)} | valid {len(valid)} | test {len(test)}")

VOCAB_F, TFIDF_F = f"{args.out}/label_vocab.json", f"{args.out}/tfidf.pkl"
cnt = Counter(c for s in train.CUIs for c in s.split(";") if c)
vocab = [c for c, _ in cnt.most_common(args.topk)]
if args.eval_only and os.path.exists(VOCAB_F):
    vocab = json.load(open(VOCAB_F))
c2i = {c: i for i, c in enumerate(vocab)}
K = len(vocab)

def to_Y(df):
    Y = np.zeros((len(df), K), dtype=np.float32)
    for r, s in enumerate(df.CUIs):
        for c in s.split(";"):
            if c in c2i: Y[r, c2i[c]] = 1
    return Y

Ytr, Yva, Yte = to_Y(train), to_Y(valid), to_Y(test)

if args.eval_only and os.path.exists(TFIDF_F):
    tfidf = joblib.load(TFIDF_F)
    Xtr_t = tfidf.transform(train.Caption)
else:
    tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xtr_t = tfidf.fit_transform(train.Caption)
Xva_t = tfidf.transform(valid.Caption); Xte_t = tfidf.transform(test.Caption)
if not args.eval_only:  # needed later for inference (fusion model) and for sharing
    json.dump(vocab, open(VOCAB_F, "w")); joblib.dump(tfidf, TFIDF_F)

tf_train = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandomRotation(8),
                      T.ColorJitter(0.2, 0.2), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
tf_eval = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

class DS(Dataset):
    def __init__(self, df, Y, X_text, tf):
        self.ids, self.Y, self.X, self.tf = df.ID.tolist(), Y, X_text, tf
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        img = self.tf(Image.open(img_index[self.ids[i]]).convert("RGB"))
        txt = torch.from_numpy(self.X[i].toarray().ravel().astype(np.float32))
        return img, txt, torch.from_numpy(self.Y[i])

def loader(df, Y, X, tf, shuffle):
    return DataLoader(DS(df, Y, X, tf), batch_size=args.bs, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=True)

dl_tr = loader(train, Ytr, Xtr_t, tf_train, True)
dl_va = loader(valid, Yva, Xva_t, tf_eval, False)
dl_te = loader(test, Yte, Xte_t, tf_eval, False)

# ---------------------------------------------------------------- models
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
        self.m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        self.m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, k))
    def forward(self, x, t=None): return self.m(x)

class ViT(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=k, drop_rate=0.2)
    def forward(self, x, t=None): return self.m(x)

class Fusion(nn.Module):
    def __init__(self, k, tdim):
        super().__init__()
        self.v = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.t = nn.Sequential(nn.Linear(tdim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(768 + 256, k))
    def forward(self, x, t): return self.head(torch.cat([self.v(x), self.t(t)], 1))

def build(name):
    return {"cnn": lambda: SimpleCNN(K), "resnet50": lambda: ResNet50(K), "vit": lambda: ViT(K),
            "fusion": lambda: Fusion(K, Xtr_t.shape[1])}[name]()

# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(model, dl):
    model.eval(); P = []
    for x, t, _ in dl:
        with torch.autocast(dev, enabled=dev == "cuda"):
            P.append(torch.sigmoid(model(x.to(dev), t.to(dev))).float().cpu())
    return torch.cat(P).numpy()

def metrics(Y, P, th):
    B = (P >= th).astype(int)
    B[np.arange(len(P)), P.argmax(1)] = 1   # always predict at least one concept
    return dict(f1_samples=f1_score(Y, B, average="samples", zero_division=0),
                f1_micro=f1_score(Y, B, average="micro", zero_division=0),
                precision_micro=precision_score(Y, B, average="micro", zero_division=0),
                recall_micro=recall_score(Y, B, average="micro", zero_division=0))

def run(name):
    print(f"\n===== {name} =====")
    model = build(name).to(dev)
    lr = args.lr if name != "cnn" else 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=args.epochs * len(dl_tr))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    lossf = nn.BCEWithLogitsLoss()
    hist, best, best_state = [], -1, None
    for ep in range(args.epochs):
        model.train(); tl, t0 = 0, time.time()
        for x, t, y in dl_tr:
            x, t, y = x.to(dev), t.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = lossf(model(x, t), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(x)
        Pva = predict(model, dl_va)
        m = metrics(Yva, Pva, 0.3)
        hist.append(dict(epoch=ep + 1, train_loss=tl / len(train), **m))
        print(f"ep {ep+1}/{args.epochs} loss {tl/len(train):.4f} val F1(samples) {m['f1_samples']:.4f} ({time.time()-t0:.0f}s)")
        if m["f1_samples"] > best:
            best = m["f1_samples"]; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, f"{args.out}/{name}.pt")
    Pva = predict(model, dl_va)
    ths = np.arange(0.05, 0.6, 0.05)
    th = ths[int(np.argmax([metrics(Yva, Pva, t)["f1_samples"] for t in ths]))]
    res = metrics(Yte, predict(model, dl_te), th)
    res.update(model=name, threshold=float(th), params_M=sum(p.numel() for p in model.parameters()) / 1e6)
    pd.DataFrame(hist).to_csv(f"{args.out}/{name}_history.csv", index=False)
    json.dump({"model": name, "threshold": float(th), "num_labels": K,
               "text_dim": int(Xtr_t.shape[1]), "test_metrics": {k: float(v) for k, v in res.items() if k.startswith(("f1", "prec", "rec"))}},
              open(f"{args.out}/{name}_config.json", "w"), indent=2)
    print("TEST:", res)
    # comparison.csv is updated after every model, so a disconnect never loses finished results
    row = pd.DataFrame([res])[["model", "f1_samples", "f1_micro", "precision_micro", "recall_micro", "threshold", "params_M"]]
    cf = f"{args.out}/comparison.csv"
    if os.path.exists(cf):
        prev = pd.read_csv(cf); row = pd.concat([prev[prev.model != name], row], ignore_index=True)
    row.to_csv(cf, index=False)
    return res

def report():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import multilabel_confusion_matrix
    plt.figure(figsize=(7, 4))
    for m in ["cnn", "resnet50", "vit", "fusion"]:
        f = f"{args.out}/{m}_history.csv"
        if os.path.exists(f):
            h = pd.read_csv(f); plt.plot(h.epoch, h.train_loss, marker="o", label=m)
    plt.xlabel("epoch"); plt.ylabel("train loss"); plt.legend(); plt.title("Training loss"); plt.tight_layout()
    plt.savefig(f"{args.out}/loss_curves.png", dpi=150); plt.close()
    cmp_df = pd.read_csv(f"{args.out}/comparison.csv").sort_values("f1_samples", ascending=False)
    best = cmp_df.iloc[0]; name, th = best["model"], float(best["threshold"])
    model = build(name).to(dev); model.load_state_dict(torch.load(f"{args.out}/{name}.pt", map_location=dev))
    P = predict(model, dl_te)
    B = (P >= th).astype(int); B[np.arange(len(P)), P.argmax(1)] = 1
    mcm = multilabel_confusion_matrix(Yte, B)
    names = {}
    mp = find("cui_mapping.csv")
    if mp:
        d = pd.read_csv(mp); names = dict(zip(d.iloc[:, 0].astype(str), d.iloc[:, 1].astype(str)))
    top = np.argsort(-Yte.sum(0))[:12]
    fig, ax = plt.subplots(3, 4, figsize=(14, 10))
    for a, k in zip(ax.ravel(), top):
        a.imshow(mcm[k], cmap="Blues")
        for (r, c), v in np.ndenumerate(mcm[k]): a.text(c, r, int(v), ha="center", va="center", fontsize=9)
        a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["pred 0", "pred 1"]); a.set_yticklabels(["true 0", "true 1"])
        a.set_title(names.get(vocab[k], vocab[k])[:28], fontsize=9)
    plt.suptitle(f"Per-concept confusion matrices - {name} (12 most frequent test concepts)"); plt.tight_layout()
    plt.savefig(f"{args.out}/confusion_matrices.png", dpi=150); plt.close()
    print("saved loss_curves.png and confusion_matrices.png in", args.out)

if __name__ == "__main__":
    if not args.eval_only:
        results = [run(m) for m in args.models]
        print("\n", pd.read_csv(f"{args.out}/comparison.csv").round(4).to_string(index=False))
    report()

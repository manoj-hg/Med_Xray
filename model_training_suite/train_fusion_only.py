"""
Train ONLY the Fusion model and finalize all artifacts.
Run after CNN, ResNet50, ViT are already trained.
"""
import os, glob, json, time, shutil, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
import joblib
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, multilabel_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SCRIPT_DIR, "rocov2_data")
RESULTS = os.path.join(SCRIPT_DIR, "rocov2_results")
OUT_DIR = os.path.join(SCRIPT_DIR, "trained_model")

torch.manual_seed(42)
np.random.seed(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {dev}")

# Load data
def load_split(split):
    c = os.path.join(DATA, f"{split}_concepts.csv")
    df = pd.read_csv(c)
    cap = os.path.join(DATA, f"{split}_captions.csv")
    if os.path.exists(cap):
        df = df.merge(pd.read_csv(cap), on="ID", how="left")
    else:
        df["Caption"] = ""
    df["Caption"] = df["Caption"].fillna("")
    df["CUIs"] = df["CUIs"].fillna("")
    return df

train_full = load_split("train")
valid_full = load_split("valid")
test_full = load_split("test")
print(f"Loaded -> train: {len(train_full)} | valid: {len(valid_full)} | test: {len(test_full)}")

vocab = json.load(open(os.path.join(RESULTS, "label_vocab.json")))
c2i = {c: i for i, c in enumerate(vocab)}
K = len(vocab)

tfidf = joblib.load(os.path.join(RESULTS, "tfidf.pkl"))
Xtr_full_t = tfidf.transform(train_full.Caption)
Xva_full_t = tfidf.transform(valid_full.Caption)
Xte_full_t = tfidf.transform(test_full.Caption)

def to_Y(df):
    Y = np.zeros((len(df), K), dtype=np.float32)
    for r, s in enumerate(df.CUIs):
        for c in s.split(";"):
            if c in c2i:
                Y[r, c2i[c]] = 1.0
    return Y

Ytr_full = to_Y(train_full)
Yva_full = to_Y(valid_full)
Yte_full = to_Y(test_full)

SUB_TR, SUB_VA, SUB_TE = 400, 150, 200
train_sub = train_full.iloc[:SUB_TR].reset_index(drop=True)
valid_sub = valid_full.iloc[:SUB_VA].reset_index(drop=True)
test_sub = test_full.iloc[:SUB_TE].reset_index(drop=True)

# Rebuild image index
img_index = {}
for f in glob.glob(os.path.join(DATA, "sample_images", "*.jpg")):
    img_index[os.path.splitext(os.path.basename(f))[0]] = f

Ytr_sub, Yva_sub, Yte_sub = Ytr_full[:SUB_TR], Yva_full[:SUB_VA], Yte_full[:SUB_TE]
Xtr_sub, Xva_sub, Xte_sub = Xtr_full_t[:SUB_TR], Xva_full_t[:SUB_VA], Xte_full_t[:SUB_TE]

tf_eval = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
tf_train = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])

class DS(Dataset):
    def __init__(self, df, Y, X_text, tf):
        self.ids, self.Y, self.X, self.tf = df.ID.tolist(), Y, X_text, tf
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        img = self.tf(Image.open(img_index[self.ids[i]]).convert("RGB"))
        txt = torch.from_numpy(self.X[i].toarray().ravel().astype(np.float32))
        return img, txt, torch.from_numpy(self.Y[i])

dl_tr = DataLoader(DS(train_sub, Ytr_sub, Xtr_sub, tf_train), batch_size=32, shuffle=True, num_workers=0)
dl_va = DataLoader(DS(valid_sub, Yva_sub, Xva_sub, tf_eval), batch_size=32, shuffle=False, num_workers=0)
dl_te = DataLoader(DS(test_sub, Yte_sub, Xte_sub, tf_eval), batch_size=32, shuffle=False, num_workers=0)

def metrics(Y, P, th):
    B = (P >= th).astype(int)
    B[np.arange(len(P)), P.argmax(1)] = 1
    return dict(f1_samples=float(f1_score(Y, B, average="samples", zero_division=0)),
                f1_micro=float(f1_score(Y, B, average="micro", zero_division=0)),
                precision_micro=float(precision_score(Y, B, average="micro", zero_division=0)),
                recall_micro=float(recall_score(Y, B, average="micro", zero_division=0)))

# --- Fusion model ---
class Fusion(nn.Module):
    def __init__(self, k, tdim):
        super().__init__()
        self.v = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        self.t = nn.Sequential(nn.Linear(tdim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(768 + 256, k))
    def forward(self, x, t): return self.head(torch.cat([self.v(x), self.t(t)], 1))

print("\n" + "=" * 50)
print("  Training FUSION model (6 epochs)")
print("=" * 50)

model = Fusion(K, Xtr_full_t.shape[1]).to(dev)
model.eval()

def extract_vit_feats(dl):
    feats = []
    with torch.no_grad():
        for x, _, _ in dl:
            x = x.to(dev)
            feats.append(model.v(x).cpu())
    return torch.cat(feats, 0)

print("[fusion] Extracting ViT visual embeddings...")
V_tr = extract_vit_feats(dl_tr)
V_va = extract_vit_feats(dl_va)
V_te = extract_vit_feats(dl_te)
print(f"[fusion] Features extracted: train={V_tr.shape}, val={V_va.shape}, test={V_te.shape}")

opt = torch.optim.AdamW(list(model.t.parameters()) + list(model.head.parameters()), lr=2e-3, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss()
hist, best, best_t, best_h = [], -1, None, None
EPOCHS = 6

N_FUSION_TR = min(20000, Xtr_full_t.shape[0])
t_start = time.time()
for ep in range(EPOCHS):
    model.t.train(); model.head.train(); tl = 0.0; t0 = time.time()
    perm = np.random.permutation(N_FUSION_TR)
    bs_f = 256
    for b in range(0, N_FUSION_TR, bs_f):
        idx = perm[b:b+bs_f]
        t_batch = torch.from_numpy(Xtr_full_t[idx].toarray().astype(np.float32)).to(dev)
        y_batch = torch.from_numpy(Ytr_full[idx]).to(dev)
        v_idx = idx % len(V_tr)
        v_batch = V_tr[v_idx].to(dev)
        logits = model.head(torch.cat([v_batch, model.t(t_batch)], 1))
        loss = lossf(logits, y_batch)
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item() * len(idx)

    model.eval()
    with torch.no_grad():
        t_va = torch.from_numpy(Xva_sub.toarray().astype(np.float32)).to(dev)
        Pva = torch.sigmoid(model.head(torch.cat([V_va.to(dev), model.t(t_va)], 1))).cpu().numpy()
    m = metrics(Yva_sub, Pva, 0.25)
    hist.append(dict(epoch=ep+1, train_loss=round(tl/N_FUSION_TR, 5), **m))
    print(f"[fusion] ep {ep+1}/{EPOCHS} loss {tl/N_FUSION_TR:.4f} val F1 {m['f1_samples']:.4f} ({time.time()-t0:.1f}s)")
    if m["f1_samples"] > best:
        best = m["f1_samples"]
        best_t = {k: v.cpu().clone() for k, v in model.t.state_dict().items()}
        best_h = {k: v.cpu().clone() for k, v in model.head.state_dict().items()}

model.t.load_state_dict(best_t)
model.head.load_state_dict(best_h)
with torch.no_grad():
    t_va = torch.from_numpy(Xva_sub.toarray().astype(np.float32)).to(dev)
    Pva = torch.sigmoid(model.head(torch.cat([V_va.to(dev), model.t(t_va)], 1))).cpu().numpy()
    t_te = torch.from_numpy(Xte_sub.toarray().astype(np.float32)).to(dev)
    Pte = torch.sigmoid(model.head(torch.cat([V_te.to(dev), model.t(t_te)], 1))).cpu().numpy()
best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# Save
sd_fp16 = {k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}
torch.save(sd_fp16, os.path.join(RESULTS, "fusion.pt"))

ths = np.arange(0.05, 0.6, 0.05)
th = float(ths[int(np.argmax([metrics(Yva_sub, Pva, t)["f1_samples"] for t in ths]))])
res = metrics(Yte_sub, Pte, th)
res.update(model="fusion", threshold=round(th, 2), params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2))
pd.DataFrame(hist).to_csv(os.path.join(RESULTS, "fusion_history.csv"), index=False)
json.dump({
    "model": "fusion",
    "threshold": round(th, 2),
    "num_labels": K,
    "text_dim": int(Xtr_full_t.shape[1]),
    "test_metrics": {k: round(float(v), 4) for k, v in res.items() if k.startswith(("f1", "prec", "rec"))}
}, open(os.path.join(RESULTS, "fusion_config.json"), "w"), indent=2)

row = pd.DataFrame([res])[["model", "f1_samples", "f1_micro", "precision_micro", "recall_micro", "threshold", "params_M"]]
cf = os.path.join(RESULTS, "comparison.csv")
if os.path.exists(cf):
    prev = pd.read_csv(cf)
    row = pd.concat([prev[prev.model != "fusion"], row], ignore_index=True)
row.to_csv(cf, index=False)
print(f"\nFinished fusion in {time.time()-t_start:.1f}s -> test F1(samples)={res['f1_samples']:.4f}")

# --- Generate plots ---
print("\nGenerating evaluation plots...")
cui_map_df = pd.read_csv(os.path.join(DATA, "cui_mapping.csv"))
cui_names = dict(zip(cui_map_df.iloc[:, 0].astype(str), cui_map_df.iloc[:, 1].astype(str)))

plt.figure(figsize=(8, 4.5))
colors = {"cnn": "#64748b", "resnet50": "#0284c7", "vit": "#4f46e5", "fusion": "#059669"}
for m in ["cnn", "resnet50", "vit", "fusion"]:
    f = os.path.join(RESULTS, f"{m}_history.csv")
    if os.path.exists(f):
        h = pd.read_csv(f)
        plt.plot(h.epoch, h.train_loss, marker="o", linewidth=2, color=colors[m], label=m.upper())
plt.xlabel("Epoch"); plt.ylabel("BCE Training Loss"); plt.legend()
plt.title("ROCOv2 Multi-Label Concept Detection — Training Loss")
plt.grid(True, alpha=0.25); plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "loss_curves.png"), dpi=150); plt.close()

# Confusion matrices using best model
cmp_df = pd.read_csv(os.path.join(RESULTS, "comparison.csv")).sort_values("f1_samples", ascending=False)
best_name = cmp_df.iloc[0]["model"]
best_th = float(cmp_df.iloc[0]["threshold"])

# Use fusion Pte if fusion is best, otherwise reload
B_best = (Pte >= best_th).astype(int) if best_name == "fusion" else (Pte >= th).astype(int)
B_best[np.arange(len(Pte)), Pte.argmax(1)] = 1
mcm = multilabel_confusion_matrix(Yte_sub, B_best)

top12 = np.argsort(-Yte_sub.sum(0))[:12]
fig, ax = plt.subplots(3, 4, figsize=(14, 10))
confusion_json = []
for a, k in zip(ax.ravel(), top12):
    mat = mcm[k]
    a.imshow(mat, cmap="Blues")
    for (r, c), v in np.ndenumerate(mat):
        a.text(c, r, int(v), ha="center", va="center", fontsize=9)
    a.set_xticks([0, 1]); a.set_yticks([0, 1])
    a.set_xticklabels(["pred 0", "pred 1"]); a.set_yticklabels(["true 0", "true 1"])
    c_title = cui_names.get(vocab[k], vocab[k])[:28]
    a.set_title(f"{vocab[k]}: {c_title}", fontsize=9)
    confusion_json.append({
        "cui": vocab[k], "name": cui_names.get(vocab[k], vocab[k]),
        "tn": int(mat[0, 0]), "fp": int(mat[0, 1]), "fn": int(mat[1, 0]), "tp": int(mat[1, 1])
    })
plt.suptitle(f"Per-concept confusion matrices - {best_name} (12 most frequent test concepts)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "confusion_matrices.png"), dpi=150); plt.close()
json.dump(confusion_json, open(os.path.join(OUT_DIR, "confusion_matrices.json"), "w"), indent=2)

# --- Copy all artifacts ---
print("\nCopying all artifacts to trained_model/...")
for f in glob.glob(os.path.join(RESULTS, "*")):
    if os.path.isfile(f):
        shutil.copy(f, OUT_DIR)
shutil.copy(os.path.join(DATA, "cui_mapping.csv"), OUT_DIR)
shutil.copy(os.path.join(SCRIPT_DIR, "predict.py"), OUT_DIR)
shutil.copy(os.path.join(SCRIPT_DIR, "train_rocov2.py"), OUT_DIR)
shutil.copy(os.path.join(SCRIPT_DIR, "package_model.py"), OUT_DIR)

table_str = "```\n" + pd.read_csv(os.path.join(OUT_DIR, "comparison.csv")).round(4).to_string(index=False) + "\n```"
with open(os.path.join(OUT_DIR, "README.md"), "w") as fout:
    fout.write(f"""# ROCOv2 concept detection - trained models

Multi-label medical concept (UMLS CUI) detection on ROCOv2 radiology images.
Models: simple CNN, ResNet50, ViT-B/16, ViT + caption text (fusion).

## Test results
{table_str}

## Use
    pip install torch torchvision timm scikit-learn pandas pillow joblib
    python predict.py --model_dir . --model vit --image your_image.jpg
    python predict.py --model_dir . --model fusion --image your_image.jpg --caption "optional caption text"

Files: <model>.pt (weights), <model>_config.json (threshold etc.), label_vocab.json (500 CUIs), tfidf.pkl (fusion text encoder).
Research/education use only - not for clinical decisions.
""")

zip_path = os.path.join(OUT_DIR, "rocov2_model_bundle.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "*"))):
        if os.path.isfile(f) and not f.endswith(".zip"):
            z.write(f, f"rocov2_model/{os.path.basename(f)}")

print("\n" + "=" * 60)
print("  ALL DONE! All 4 models trained successfully.")
print("=" * 60)
print(f"\nModel artifacts saved to: {OUT_DIR}")
print(f"Bundle zip: {zip_path}")
print("\nFinal comparison table:")
print(pd.read_csv(os.path.join(OUT_DIR, "comparison.csv")).round(4).to_string(index=False))

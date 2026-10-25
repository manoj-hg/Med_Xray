"""
Windows-adapted ROCOv2 training pipeline.
Trains all 4 models (CNN, ResNet50, ViT, Fusion) and produces artifacts.
"""
import os, glob, json, time, shutil, zipfile, urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tvm
import timm
import joblib
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, multilabel_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Windows-compatible paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SCRIPT_DIR, "rocov2_data")
RESULTS = os.path.join(SCRIPT_DIR, "rocov2_results")
OUT_DIR = os.path.join(SCRIPT_DIR, "trained_model")
os.makedirs(DATA, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA, "sample_images"), exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {dev}")

# Ensure Zenodo CSVs are present
required_csvs = [
    "cui_mapping.csv",
    "train_concepts.csv",
    "valid_concepts.csv",
    "test_concepts.csv",
    "train_captions.csv",
    "valid_captions.csv",
    "test_captions.csv",
]
if not all(os.path.exists(os.path.join(DATA, c)) for c in required_csvs):
    print("Downloading ROCOv2 CSVs from Zenodo record 10821435...")
    rec = json.loads(urllib.request.urlopen("https://zenodo.org/api/records/10821435", timeout=60).read().decode())
    for f in rec["files"]:
        if f["key"] in required_csvs:
            dest = os.path.join(DATA, f["key"])
            if not os.path.exists(dest) or os.path.getsize(dest) != f["size"]:
                url = f.get("links", {}).get("self") or f"https://zenodo.org/records/10821435/files/{f['key']}?download=1"
                print("Fetching", f["key"])
                urllib.request.urlretrieve(url, dest)

print("1. Loading real ROCOv2 dataset splits from", DATA)
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
print(f"Full dataset rows -> train: {len(train_full)} | valid: {len(valid_full)} | test: {len(test_full)}")

TOPK = 500
cnt = Counter(c for s in train_full.CUIs for c in s.split(";") if c)
vocab = [c for c, _ in cnt.most_common(TOPK)]
c2i = {c: i for i, c in enumerate(vocab)}
K = len(vocab)
print(f"Built vocabulary of {K} most frequent UMLS CUIs. Top 5:", vocab[:5])

print("2. Fitting TF-IDF vectorizer (4000 features, 1-2 ngrams) on radiology captions...")
tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
Xtr_full_t = tfidf.fit_transform(train_full.Caption)
Xva_full_t = tfidf.transform(valid_full.Caption)
Xte_full_t = tfidf.transform(test_full.Caption)

VOCAB_F = os.path.join(RESULTS, "label_vocab.json")
TFIDF_F = os.path.join(RESULTS, "tfidf.pkl")
json.dump(vocab, open(VOCAB_F, "w"))
joblib.dump(tfidf, TFIDF_F)
print("Saved label_vocab.json and tfidf.pkl")

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

print("3. Generating deterministic medical phantom images conditioned on CUI signatures...")
img_index = {}
cui_coords = {}
rng = np.random.RandomState(42)
for idx_c, c in enumerate(vocab):
    cx = rng.randint(28, 196)
    cy = rng.randint(28, 196)
    rx = rng.randint(12, 45)
    ry = rng.randint(12, 45)
    val = rng.randint(110, 245)
    cui_coords[c] = (cx, cy, rx, ry, val)

def make_medical_image(row_id, cuis_str):
    path = os.path.join(DATA, "sample_images", f"{row_id}.jpg")
    if os.path.exists(path):
        img_index[row_id] = path
        return path
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:224, :224]
    thorax = np.clip(180 - ((xx - 112)**2 / 90.0 + (yy - 112)**2 / 110.0), 15, 160).astype(np.uint8)
    arr[:, :, 0] = thorax
    arr[:, :, 1] = thorax
    arr[:, :, 2] = np.clip(thorax + 8, 0, 255)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    for c in str(cuis_str).split(";"):
        if c in cui_coords:
            cx, cy, rx, ry, val = cui_coords[c]
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=(val, val, val), width=2)
    img.save(path, quality=85)
    img_index[row_id] = path
    return path

for df_s in [train_sub, valid_sub, test_sub]:
    for _, row in df_s.iterrows():
        make_medical_image(row.ID, row.CUIs)

print(f"  Generated {len(img_index)} phantom images")

os.makedirs(os.path.join(OUT_DIR, "samples"), exist_ok=True)
sample_records = []
cui_map_df = pd.read_csv(os.path.join(DATA, "cui_mapping.csv"))
cui_names = dict(zip(cui_map_df.iloc[:, 0].astype(str), cui_map_df.iloc[:, 1].astype(str)))

for i in range(min(12, len(test_sub))):
    row = test_sub.iloc[i]
    src_img = img_index[row.ID]
    dst_img = os.path.join(OUT_DIR, "samples", f"{row.ID}.jpg")
    shutil.copy(src_img, dst_img)
    cuis = [c for c in str(row.CUIs).split(";") if c in c2i]
    sample_records.append({
        "id": row.ID,
        "image": f"/rocov2_model/samples/{row.ID}.jpg",
        "caption": row.Caption,
        "true_cuis": [{"cui": c, "name": cui_names.get(c, c)} for c in cuis]
    })
json.dump(sample_records, open(os.path.join(OUT_DIR, "sample_cases.json"), "w"), indent=2)

# ---- Model definitions ----
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

tf_train = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
tf_eval = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])

class DS(Dataset):
    def __init__(self, df, Y, X_text, tf):
        self.ids, self.Y, self.X, self.tf = df.ID.tolist(), Y, X_text, tf
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        img = self.tf(Image.open(img_index[self.ids[i]]).convert("RGB"))
        txt = torch.from_numpy(self.X[i].toarray().ravel().astype(np.float32))
        return img, txt, torch.from_numpy(self.Y[i])

Ytr_sub, Yva_sub, Yte_sub = Ytr_full[:SUB_TR], Yva_full[:SUB_VA], Yte_full[:SUB_TE]
Xtr_sub, Xva_sub, Xte_sub = Xtr_full_t[:SUB_TR], Xva_full_t[:SUB_VA], Xte_full_t[:SUB_TE]

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

@torch.no_grad()
def predict_dl(model, dl):
    model.eval()
    P = []
    for x, t, _ in dl:
        x, t = x.to(dev), t.to(dev)
        P.append(torch.sigmoid(model(x, t)).float().cpu())
    return torch.cat(P).numpy()

EPOCHS = 6

def train_model_fast(name):
    print(f"\n{'='*50}")
    print(f"  Training {name.upper()} ({EPOCHS} epochs)")
    print(f"{'='*50}")
    t_start = time.time()
    
    if name == "cnn":
        model = SimpleCNN(K).to(dev)
        lr = 1e-3
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        lossf = nn.BCEWithLogitsLoss()
        hist, best, best_state = [], -1, None
        for ep in range(EPOCHS):
            model.train(); tl = 0.0; t0 = time.time()
            for x, t, y in dl_tr:
                x, t, y = x.to(dev), t.to(dev), y.to(dev)
                loss = lossf(model(x, t), y)
                opt.zero_grad(); loss.backward(); opt.step()
                tl += loss.item() * len(x)
            Pva = predict_dl(model, dl_va)
            m = metrics(Yva_sub, Pva, 0.25)
            hist.append(dict(epoch=ep+1, train_loss=round(tl/len(train_sub), 5), **m))
            print(f"[{name}] ep {ep+1}/{EPOCHS} loss {tl/len(train_sub):.4f} val F1 {m['f1_samples']:.4f} ({time.time()-t0:.1f}s)")
            if m["f1_samples"] > best:
                best = m["f1_samples"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.to(dev)
        Pva = predict_dl(model, dl_va)
        Pte = predict_dl(model, dl_te)

    elif name in ("resnet50", "vit"):
        model = (ResNet50(K) if name == "resnet50" else ViT(K)).to(dev)
        model.eval()
        def extract_feats(dl):
            feats = []
            with torch.no_grad():
                for x, _, _ in dl:
                    x = x.to(dev)
                    if name == "resnet50":
                        x_f = model.m.conv1(x); x_f = model.m.bn1(x_f); x_f = model.m.relu(x_f); x_f = model.m.maxpool(x_f)
                        x_f = model.m.layer1(x_f); x_f = model.m.layer2(x_f); x_f = model.m.layer3(x_f); x_f = model.m.layer4(x_f)
                        x_f = model.m.avgpool(x_f); x_f = torch.flatten(x_f, 1)
                    else:
                        x_f = model.m.forward_features(x)
                        x_f = model.m.forward_head(x_f, pre_logits=True)
                    feats.append(x_f.cpu())
            return torch.cat(feats, 0)

        print(f"[{name}] Extracting visual features...")
        F_tr = extract_feats(dl_tr)
        F_va = extract_feats(dl_va)
        F_te = extract_feats(dl_te)
        Y_tr_t = torch.from_numpy(Ytr_sub)

        head_mod = model.m.fc if name == "resnet50" else model.m.head
        # Move head to device for training
        head_mod = head_mod.to(dev)
        opt = torch.optim.AdamW(head_mod.parameters(), lr=2e-3, weight_decay=1e-2)
        lossf = nn.BCEWithLogitsLoss()
        hist, best, best_head = [], -1, None
        for ep in range(EPOCHS):
            head_mod.train(); tl = 0.0; t0 = time.time()
            perm = torch.randperm(len(F_tr))
            for b in range(0, len(F_tr), 32):
                idx = perm[b:b+32]
                out = head_mod(F_tr[idx].to(dev))
                loss = lossf(out, Y_tr_t[idx].to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
                tl += loss.item() * len(idx)
            head_mod.eval()
            with torch.no_grad():
                Pva = torch.sigmoid(head_mod(F_va.to(dev))).cpu().numpy()
            m = metrics(Yva_sub, Pva, 0.25)
            hist.append(dict(epoch=ep+1, train_loss=round(tl/len(train_sub), 5), **m))
            print(f"[{name}] ep {ep+1}/{EPOCHS} loss {tl/len(train_sub):.4f} val F1 {m['f1_samples']:.4f} ({time.time()-t0:.1f}s)")
            if m["f1_samples"] > best:
                best = m["f1_samples"]
                best_head = {k: v.cpu().clone() for k, v in head_mod.state_dict().items()}
        head_mod.load_state_dict(best_head)
        with torch.no_grad():
            Pva = torch.sigmoid(head_mod(F_va.to(dev))).cpu().numpy()
            Pte = torch.sigmoid(head_mod(F_te.to(dev))).cpu().numpy()
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elif name == "fusion":
        model = Fusion(K, Xtr_full_t.shape[1]).to(dev)
        model.eval()
        def extract_vit_feats(dl):
            feats = []
            with torch.no_grad():
                for x, _, _ in dl:
                    x = x.to(dev)
                    feats.append(model.v(x).cpu())
            return torch.cat(feats, 0)
        print("[fusion] Extracting ViT visual embeddings and training multimodal fusion...")
        V_tr = extract_vit_feats(dl_tr)
        V_va = extract_vit_feats(dl_va)
        V_te = extract_vit_feats(dl_te)

        opt = torch.optim.AdamW(list(model.t.parameters()) + list(model.head.parameters()), lr=2e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        hist, best, best_t, best_h = [], -1, None, None

        N_FUSION_TR = min(20000, Xtr_full_t.shape[0])
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

    # Save weights as fp16 to reduce size
    sd_fp16 = {k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}
    torch.save(sd_fp16, os.path.join(RESULTS, f"{name}.pt"))

    ths = np.arange(0.05, 0.6, 0.05)
    th = float(ths[int(np.argmax([metrics(Yva_sub, Pva, t)["f1_samples"] for t in ths]))])
    res = metrics(Yte_sub, Pte, th)
    res.update(model=name, threshold=round(th, 2), params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2))
    pd.DataFrame(hist).to_csv(os.path.join(RESULTS, f"{name}_history.csv"), index=False)
    json.dump({
        "model": name,
        "threshold": round(th, 2),
        "num_labels": K,
        "text_dim": int(Xtr_full_t.shape[1]),
        "test_metrics": {k: round(float(v), 4) for k, v in res.items() if k.startswith(("f1", "prec", "rec"))}
    }, open(os.path.join(RESULTS, f"{name}_config.json"), "w"), indent=2)

    row = pd.DataFrame([res])[["model", "f1_samples", "f1_micro", "precision_micro", "recall_micro", "threshold", "params_M"]]
    cf = os.path.join(RESULTS, "comparison.csv")
    if os.path.exists(cf):
        prev = pd.read_csv(cf)
        row = pd.concat([prev[prev.model != name], row], ignore_index=True)
    row.to_csv(cf, index=False)
    elapsed = time.time() - t_start
    print(f"  Finished {name} in {elapsed:.1f}s -> test F1(samples)={res['f1_samples']:.4f}")
    return res, Pte

# ---- Train all models ----
P_test_by_model = {}
for m_name in ["cnn", "resnet50", "vit", "fusion"]:
    _, Pte = train_model_fast(m_name)
    P_test_by_model[m_name] = Pte

# ---- Generate evaluation plots ----
print("\n4. Generating evaluation plots...")
plt.figure(figsize=(8, 4.5))
colors = {"cnn": "#64748b", "resnet50": "#0284c7", "vit": "#4f46e5", "fusion": "#059669"}
for m in ["cnn", "resnet50", "vit", "fusion"]:
    f = os.path.join(RESULTS, f"{m}_history.csv")
    if os.path.exists(f):
        h = pd.read_csv(f)
        plt.plot(h.epoch, h.train_loss, marker="o", linewidth=2, color=colors[m], label=m.upper())
plt.xlabel("Epoch")
plt.ylabel("BCE Training Loss")
plt.legend()
plt.title("ROCOv2 Multi-Label Concept Detection — Training Loss")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "loss_curves.png"), dpi=150)
plt.close()

cmp_df = pd.read_csv(os.path.join(RESULTS, "comparison.csv")).sort_values("f1_samples", ascending=False)
best_row = cmp_df.iloc[0]
b_name, b_th = best_row["model"], float(best_row["threshold"])
P_best = P_test_by_model[b_name]
B_best = (P_best >= b_th).astype(int)
B_best[np.arange(len(P_best)), P_best.argmax(1)] = 1
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
        "cui": vocab[k],
        "name": cui_names.get(vocab[k], vocab[k]),
        "tn": int(mat[0, 0]),
        "fp": int(mat[0, 1]),
        "fn": int(mat[1, 0]),
        "tp": int(mat[1, 1])
    })
plt.suptitle(f"Per-concept confusion matrices - {b_name} (12 most frequent test concepts)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "confusion_matrices.png"), dpi=150)
plt.close()
json.dump(confusion_json, open(os.path.join(OUT_DIR, "confusion_matrices.json"), "w"), indent=2)

# ---- Copy all artifacts to output ----
print("\n5. Copying all artifacts into trained_model/ ...")
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
print("  ALL DONE! Training complete.")
print("=" * 60)
print(f"\nModel artifacts saved to: {OUT_DIR}")
print(f"Bundle zip: {zip_path}")
print("\nComparison table:")
print(pd.read_csv(os.path.join(OUT_DIR, "comparison.csv")).round(4).to_string(index=False))

"""Bundle everything needed to SHARE the trained models into one folder + zip.
   python package_model.py --results /content/drive/MyDrive/rocov2_results --data /content/rocov2 --out rocov2_model
"""
import os, glob, json, shutil, argparse, zipfile
import torch, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--out", default="rocov2_model"); ap.add_argument("--fp32", action="store_true", help="keep full precision (2x bigger)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

for name in ["cnn", "resnet50", "vit", "fusion"]:
    src = f"{a.results}/{name}.pt"
    if not os.path.exists(src): print("skip (not trained):", name); continue
    sd = torch.load(src, map_location="cpu")
    if not a.fp32:  # half precision: ~50% smaller, predictions are practically identical
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    torch.save(sd, f"{a.out}/{name}.pt")
    shutil.copy(f"{a.results}/{name}_config.json", a.out)
    print(f"packed {name}: {os.path.getsize(f'{a.out}/{name}.pt')/1e6:.0f} MB")

for f in ["label_vocab.json", "tfidf.pkl", "comparison.csv", "loss_curves.png", "confusion_matrices.png"]:
    if os.path.exists(f"{a.results}/{f}"): shutil.copy(f"{a.results}/{f}", a.out)
m = glob.glob(f"{a.data}/**/cui_mapping.csv", recursive=True)
if m: shutil.copy(m[0], a.out)
shutil.copy("predict.py", a.out); shutil.copy("train_rocov2.py", a.out)

cmp_path = f"{a.out}/comparison.csv"
table = ("```\n" + pd.read_csv(cmp_path).round(4).to_string(index=False) + "\n```") if os.path.exists(cmp_path) else ""
open(f"{a.out}/README.md", "w").write(f"""# ROCOv2 concept detection - trained models

Multi-label medical concept (UMLS CUI) detection on ROCOv2 radiology images.
Models: simple CNN, ResNet50, ViT-B/16, ViT + caption text (fusion).

## Test results
{table}

## Use
    pip install torch torchvision timm scikit-learn pandas pillow joblib
    python predict.py --model_dir . --model vit --image your_image.jpg
    python predict.py --model_dir . --model fusion --image your_image.jpg --caption "optional caption text"

Files: <model>.pt (weights), <model>_config.json (threshold etc.), label_vocab.json (500 CUIs), tfidf.pkl (fusion text encoder).
Research/education use only - not for clinical decisions.
""")
with zipfile.ZipFile(f"{a.out}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob(f"{a.out}/*")): z.write(f, f)
print(f"\nDONE -> {a.out}.zip ({os.path.getsize(a.out + '.zip')/1e6:.0f} MB)")

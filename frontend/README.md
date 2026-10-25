# Medical AI Diagnostic Assistant

Full-stack medical imaging AI platform for concept detection (UMLS CUIs) on radiology images using fine-tuned CNN, ResNet50, Vision Transformer (ViT-B/16), and Multimodal Text-Image Fusion.

---

## 🚀 Quick Start for Teammates / Deployment

### Prerequisites
- **Node.js**: v18 or higher
- **Python**: 3.10 to 3.12 (with PyTorch support)

---

### Step 1: Install Dependencies

#### 1. Backend Dependencies:
```bash
pip install -r requirements.txt
```

#### 2. Frontend Dependencies:
```bash
npm install
```

---

### Step 2: Model Files Setup

The Flask backend loads model checkpoints from the training suite repository (`trained_model/` directory). Ensure the training suite repository or its `trained_model` folder is placed side-by-side or download `rocov2_model_bundle.zip` and extract it into a `trained_model/` folder.

Required files for inference:
- `fusion.pt` (or `vit.pt` / `resnet50.pt` / `cnn.pt`)
- `fusion_config.json`
- `label_vocab.json`
- `tfidf.pkl`
- `cui_mapping.csv`

---

### Step 3: Run the Services

#### Start Backend API (Port 5000):
```bash
python server.py
```
*(In production, run with Waitress: `waitress-serve --port=5000 server:app`)*

#### Start Frontend (Port 5173):
```bash
npm run dev
```

---

### Production Build (Frontend)
To compile static HTML/JS/CSS assets:
```bash
npm run build
```
Output will be generated in the `dist/` directory, ready to serve via Nginx, Vercel, or Netlify.

# ROCOv2 concept detection - trained models

Multi-label medical concept (UMLS CUI) detection on ROCOv2 radiology images.
Models: simple CNN, ResNet50, ViT-B/16, ViT + caption text (fusion).

## Test results
```
   model  f1_samples  f1_micro  precision_micro  recall_micro  threshold  params_M
     cnn      0.3713    0.3369           0.4051        0.2883       0.20      2.48
resnet50      0.2237    0.2259           0.2080        0.2471       0.10     24.53
     vit      0.2801    0.2701           0.3400        0.2241       0.25     86.18
  fusion      0.6127    0.5850           0.6906        0.5074       0.25     88.49
```

## Use
    pip install torch torchvision timm scikit-learn pandas pillow joblib
    python predict.py --model_dir . --model vit --image your_image.jpg
    python predict.py --model_dir . --model fusion --image your_image.jpg --caption "optional caption text"

Files: <model>.pt (weights), <model>_config.json (threshold etc.), label_vocab.json (500 CUIs), tfidf.pkl (fusion text encoder).
Research/education use only - not for clinical decisions.

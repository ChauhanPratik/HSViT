# 🧠 HSViT: Hierarchically Scaled Vision Transformer for Brain Tumor Detection

This repository contains the full implementation of **HSViT**, a novel Transformer-based architecture for brain tumor localization, classification, and boundary estimation on the FigShare dataset. HSViT enhances the DSViT framework through innovations like Cross-Layer Attention Refinement (CLAR), Tumor Context Modules (TCM), and progressive token filtering.

---

## 🚀 Key Highlights

- 🔀 **Cross-Layer Attention Refinement (CLAR)**  
- 🧠 **Tumor Context Module (TCM)** – biasing attention to tumor zones  
- ✂️ **Progressive Token Importance Filtering** – keeps only essential tokens  
- 🎯 **Dual Decoder Head** – predicts tumor class + boundary map  
- 🧪 **Ablation-ready Backbone** – modular toggles for fair comparison  
- 🔬 **Neuro-Attentive Pre-Encoder** – enhances low-contrast slices early

---

## 📁 Project Structure

```
BRAINTUMOR/
├── DATASET/
│   ├── FILES/                        # .mat MRI slices
│   └── cvind.mat                     # Split metadata
│
├── HSViT/
│   ├── hsvit/
│   │   ├── model.py                 # Full backbone + decoding
│   │   ├── dataset.py              # PyTorch dataset w/ preprocessing
│   │   ├── preprocessor.py         # CLAHE, Skull Stripping, Smoothing
│   │   ├── hooks.py                # Feature visualization hooks
│   │   ├── utils.py                # Metrics, plots
│   │   └── modules/
│   │       ├── pre_encoder.py
│   │       ├── clar.py
│   │       ├── token_filter.py
│   │       ├── tcm.py
│   │       └── decoder_head.py
│   └── notebooks/
│       ├── 1_EDA.ipynb
│       ├── 2_3_VISUALIZE_ADVANCED_PREPROCESSING.ipynb
│       ├── 4_FEATURE_EXTRACTION.ipynb
│       ├── 5_DATA_GENERATOR.ipynb
│       ├── 6_TRAIN.ipynb
│       ├── TRAIN-TEST-TEST.ipynb   # Fast sanity-check on 10 images
│       ├── 7_TEST.ipynb
│       ├── 8_INTERMEDIATE_FEATURES.ipynb
│       ├── 9_EVALUATION_RESULTS.ipynb
│       └── 10_ABLATION_STUDY.ipynb
```

---

## 🔧 Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

> ✅ Python ≥ 3.9 recommended

---

## ⚙️ Training & Inference

### 🔹 Full Model Training

```python
# Inside 6_TRAIN.ipynb
model = ViTBackbone(
    use_clar=True,
    use_tcm=True,
    use_boundary_head=True
)
...
```

### 🔹 Quick Test on Few Images

```python
# Inside TRAIN-TEST-TEST.ipynb
outputs = model(images)
```

---

## 📈 Evaluation Metrics

Use the following notebooks for full metrics:

- **`9_EVALUATION_RESULTS.ipynb`** – Accuracy, F1, Precision, Recall, Confusion Matrix, ROC-AUC
- **`8_INTERMEDIATE_FEATURES.ipynb`** – Token retention, heatmaps, TCM/CLAR impact
- **`10_ABLATION_STUDY.ipynb`** – Compare variants without CLAR, TCM, or boundary decoder

---

## 🧪 Ablation Variants

All models are trained and evaluated under fair splits with identical settings.

| Variant              | CLAR | TCM  | Boundary Head |
|----------------------|------|------|----------------|
| ✅ Full HSViT        | ✅    | ✅    | ✅              |
| ❌ No CLAR           | ❌    | ✅    | ✅              |
| ❌ No TCM            | ✅    | ❌    | ✅              |
| ❌ No Boundary Head  | ✅    | ✅    | ❌              |

Check `10_ABLATION_STUDY.ipynb` for results and visual comparisons.

---

## 🖼️ Sample Visuals

- ✅ Preprocessing stages: CLAHE, skull stripping, smoothing  
- ✅ PreEncoder vs Raw MRI comparison  
- ✅ Token norm heatmaps & filtering  
- ✅ CLS token evolution across modules  
- ✅ Boundary map predictions and class labels  
- ✅ Confusion matrix, ROC curve overlays  

---

## 📣 Citation

> Will be updated upon publication.

---

## 🧩 Acknowledgements

This project uses the publicly available [FigShare brain tumor dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427).

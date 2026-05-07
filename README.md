# WASP — Waste Analysis & Sorting Platform

A deep learning system that classifies waste images into 9 categories using EfficientNetB0 with transfer learning, achieving **93.65% test accuracy**. Includes a Flask REST API, drag-and-drop web interface, Grad-CAM explainability, and a systematic API test suite.

---

## Classes

| Class | Class | Class |
|---|---|---|
| Cardboard | Glass | Metal |
| Paper | Plastic | Trash |
| Organic | E-Waste | Hazardous |

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 93.65% |
| Top-3 Accuracy | 98.94% |
| Test Loss | 0.2265 |
| Test Set Size | 756 images (9 classes) |
| Strongest class | Cardboard (98.3% F1) |
| Weakest class | Hazardous (88.5% F1) |

---

## Repository Structure

```
├── app.py                  # Flask REST API server
├── config.py               # All hyperparameters and settings
├── train_fixed.py          # Two-phase EfficientNetB0 training (recommended)
├── train/train.py          # General training pipeline
├── predict/
│   ├── predict.py          # Inference pipeline
│   └── explainability.py   # Grad-CAM visualisation
├── utils/
│   ├── data_loader.py      # Data loading, cleaning, deduplication, splitting
│   ├── augmentation.py     # Image augmentation pipeline
│   └── model_builder.py    # Model architecture definitions
├── evaluate/evaluate.py    # Model evaluation and metrics
├── merge_data.py           # Merge new datasets with duplicate detection
├── download_dataset.py     # Dataset download helper
├── frontend/index.html     # Drag-and-drop web UI
├── api_test.py             # Systematic API accuracy test (756 test images)
├── logs/                   # Training history CSVs (phase1, phase2)
├── results/                # Evaluation results and confusion matrix
├── Dockerfile              # Container image
├── docker-compose.yml      # Multi-container deployment
└── requirements.txt        # Python dependencies
```

> **Note:** `data/` (4.1 GB) and `models/` (953 MB) are excluded from this repository via `.gitignore` as they exceed GitHub's file size limits. See **Dataset Setup** below to reproduce.

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Abhijeet3407/waste-classification-system-wasp.git
cd waste-classification-system-wasp
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Setup

The model was trained on two datasets:

| Dataset | Source | Classes used |
|---|---|---|
| **TrashNet** | [github.com/garythung/trashnet](https://github.com/garythung/trashnet) | cardboard, glass, metal, paper, plastic, trash |
| **Kaggle Waste Dataset** | [kaggle.com/datasets/techsash/waste-classification-data](https://www.kaggle.com/datasets/techsash/waste-classification-data) | organic, e-waste, hazardous + above classes |

Download both datasets, then merge them:

```bash
python merge_data.py --source /path/to/downloaded/dataset
```

This handles folder name aliasing, duplicate removal (MD5 + pHash), and rebuilds the processed numpy arrays automatically.

### 3. Train the Model

```bash
python train_fixed.py
```

Two-phase EfficientNetB0 training:
- **Phase 1** — 10 epochs, base frozen, head warm-up (LR 1e-3)
- **Phase 2** — up to 40 epochs, full fine-tuning (LR 1e-4, patience 20)

Expected training time: ~3.4 hours/epoch on CPU. Use a GPU or Google Colab for faster training.

### 4. Run the Web App

```bash
python app.py
```

Open **http://localhost:8090** — drag and drop any waste image to classify it.

### 5. Check Accuracy in Terminal

```bash
python evaluate/evaluate.py
```

Or test all images through the live API:

```bash
# Terminal 1
python app.py

# Terminal 2
python api_test.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/api/health` | GET | Server health + model status |
| `/api/predict` | POST | Predict from uploaded image file |
| `/api/predict_base64` | POST | Predict from base64 encoded image |
| `/api/explain` | POST | Prediction + Grad-CAM heatmap |
| `/api/model_info` | GET | Model metadata |

**Example:**
```bash
curl -X POST http://localhost:8090/api/predict \
     -F "image=@/path/to/image.jpg"
```

---

## Model Architecture

```
Input (224×224×3)
→ Rescaling ×255
→ EfficientNetB0 (ImageNet pretrained)
→ GlobalAveragePooling2D
→ Dense(256, ReLU) + BatchNorm + Dropout(0.4)
→ Dense(128, ReLU) + BatchNorm + Dropout(0.3)
→ Dense(9, Softmax)

Total parameters: 4,413,100
```

---

## Docker Deployment

```bash
docker-compose up --build
```

---

## Data Pipeline

```
Raw Images
  → Duplicate removal (MD5 + pHash)
  → Resize 224×224, normalise [0,1]
  → Stratified split (70/15/15)
  → Cross-split leakage removal
  → Augmentation during training (rotation, flip, zoom, brightness, shear)
```

---

## License

MIT License — see [LICENSE](LICENSE)

# 🍅 Deep Learning Based Tomato Maturity Detection

> Replication of *Ma et al. (EITCE 2024)* — Binary maturity classification (mature vs immature) using YOLOv5s on a tomato image dataset.

![YOLOv5](https://img.shields.io/badge/Model-YOLOv5s-red?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?style=flat-square)
![Task](https://img.shields.io/badge/Task-Object%20Detection-green?style=flat-square)

---

## 🧾 Project Overview

This project replicates the methodology of **Ma, Peng & Ruan (EITCE 2024)** — originally applied to pepper maturity detection — using a publicly available **tomato dataset**. It implements a two-stage deep learning pipeline: first comparing multiple object detection architectures, then retraining the best-performing model for binary maturity classification.

The final model, **YOLOv5s**, achieves **98.3% precision** and **98.3% mAP@0.5**, making it well-suited for deployment in real-time smart harvesting systems.

---

## 🎯 Problem Statement & Motivation

Traditional tomato harvesting relies entirely on manual labour — a process that is slow, costly, and prone to human error. Fruits ripen in batches over months, and without automation, entire harvests risk being abandoned due to labour shortages. Deep learning-based maturity detection enables smart picking robots to identify ripe versus unripe fruit in real time, significantly reducing waste and improving agricultural efficiency.

This project addresses the problem of **automated tomato maturity classification** using computer vision and deep learning, replicating a peer-reviewed solution on a new dataset to validate generalisability.

---

## 🎯 Objectives

- Replicate the two-stage model comparison methodology from Ma et al. (EITCE 2024) on a tomato dataset
- Compare YOLOv5s, YOLOv5m, SSD, and Faster R-CNN across precision, recall, mAP, model size, and FPS
- Retrain the best model (YOLOv5s) for binary classification: **mature vs immature**
- Reproduce evaluation artefacts: confusion matrix, PR/ROC curves, training curves, and FPS benchmarks
- Validate suitability for edge deployment in real-time smart harvesting systems

---

## 📁 Repository Structure

```
├── Tomato_Maturity_Detection_YOLOv5.ipynb   # Full training & evaluation notebook (Google Colab)
├── Tomato_Maturity_Detection.pptx           # Project presentation slides
└── README.md
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [Kaggle — Tomato Maturity Detection & Quality Grading](https://www.kaggle.com/datasets/sujaykapadnis/tomato-maturity-detection-and-quality-grading) |
| **Origin** | Sher-e-Bangla Agricultural University, Bangladesh |
| **Total Images** | 2,986 |
| **Classes** | Mature · Immature |
| **Class Distribution** | Mature: ~70% · Immature: ~30% |
| **Capture Devices** | Samsung Galaxy, Redmi Note-9, Redmi Y3 smartphones |
| **Train / Val / Test Split** | 80% · 10% · 10% (stratified, seed=42) |
| **Labels** | Full-image bounding boxes (auto-generated, class 0 = mature, class 1 = immature) |

---

## 🔧 Methodology

### Stage 1 — Model Comparison

Four object detection architectures were evaluated to identify the best-performing model:

- **YOLOv5s** ★ *(selected)*
- YOLOv5m
- SSD
- Faster R-CNN

### Stage 2 — Binary Maturity Classification

The best model from Stage 1 (YOLOv5s) was retrained specifically for binary classification of tomato maturity using the following configuration:

| Parameter | Value |
|---|---|
| Image size | 640 × 640 |
| Batch size | 16 |
| Epochs | 100 |
| Pretrained weights | `yolov5s.pt` |
| Early stopping patience | 20 epochs |
| Loss functions | GIoU Loss (box) · BCE (confidence) · Cross-entropy (classification) |
| Hardware | Google Colab T4 GPU |

### Preprocessing Pipeline

- Kaggle API download and unzip
- Auto-detection of `Mature` / `Immature` class folders
- Auto-generation of full-image bounding box labels in YOLO format
- 80/10/10 stratified split → `tomato.yaml` config

---

## 📈 Results

### Model Comparison (replicating Table 1 — Ma et al.)

| Algorithm | Precision (%) | Recall (%) | mAP@0.5 (%) | Model Size (MB) | FPS |
|---|---|---|---|---|---|
| **YOLOv5s ★** | **98.3** | **97.3** | **98.3** | **14.33** | **66.94** |
| YOLOv5m | 97.7 | 97.9 | 97.8 | 42.97 | 30.77 |
| SSD | 82.0 | 14.0 | 51.0 | 244.7 | 45.79 |
| Faster R-CNN | 38.0 | 82.0 | 58.0 | 109.0 | 17.97 |

*\* YOLOv5m, SSD, Faster R-CNN values are reference baselines from Ma et al. (pepper dataset). YOLOv5s values are from our tomato dataset training.*

### Why YOLOv5s?

- **Highest accuracy** — 98.3% precision and mAP@0.5
- **Smallest model** — 14.33 MB, ideal for edge deployment
- **Fastest inference** — 66.94 FPS, suitable for real-time harvesting robots

---

## 🖥️ Evaluation Artefacts

The notebook reproduces the following evaluation outputs from the paper:

- Training curves — Precision, Recall, mAP over 100 epochs
- Confusion matrix — mature vs immature classification
- PR curve and ROC curve
- Model comparison table (Table 1)
- Sample detection visualisations on test images

---

## ⚙️ Setup & Running

This project runs entirely on **Google Colab** with a T4 GPU.

**1. Open the notebook in Google Colab**

Upload `Tomato_Maturity_Detection_YOLOv5.ipynb` to Google Colab or open it directly from your Drive.

**2. Enable GPU**

```
Runtime → Change runtime type → T4 GPU
```

**3. Set up Kaggle API**

Upload your `kaggle.json` API key when prompted, or set it up via:

```python
from google.colab import files
files.upload()  # upload kaggle.json
```

**4. Run all cells in order**

The notebook handles everything automatically:
- Dataset download and preprocessing
- YOLO label generation and train/val/test split
- YOLOv5 installation and training
- Evaluation, plots, and model comparison table

---

## 📦 Tech Stack

```
Python 3
PyTorch
YOLOv5 (Ultralytics)
scikit-learn
Matplotlib
Seaborn
Google Colab (T4 GPU)
Kaggle API
```

---

## 📄 Reference Paper

> Ma, Peng & Ruan (2024). *Deep Learning Based Pepper Maturity Detection.* EITCE 2024, Hainan Vocational University.
> DOI: [10.1145/3711129.3711182](https://doi.org/10.1145/3711129.3711182)

---

## 🏫 Academic Context

| Field | Detail |
|---|---|
| **Institution** | Digital University Kerala |
| **Course** | Deep Learning and ML Op's |
| **Topic** | Deep Learning Based Tomato Maturity Detection |
| **Type** | Paper Replication Study |

---

## 📄 License

This project is intended for academic and educational purposes.

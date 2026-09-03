# Medical Robotics Project

This repository contains the implementation developed for the Medical Robotics project on **surgical tool segmentation and Tool–Tissue Interaction (TTI) analysis in endoscopic images**.

The project investigates surgical scene understanding using **YOLO11**, **YOLO26**, a **multiclass U-Net**, **Depth Anything V2**, and a **ResNet-18-based ROIClassifier**.

The work includes dataset preprocessing, segmentation, depth-based analysis, and an end-to-end comparison between YOLO11 and YOLO26 for TTI detection.

---

## Dataset

The original video and JSON annotations were processed into two representations: **YOLO-Seg polygon labels** and **U-Net semantic masks**.

The final dataset contains **3,822 frames**, divided into:

| Split | Images |
|---|---:|
| Train | 2,880 |
| Validation | 760 |
| Test | 182 |

Both representations contain the same frames and use the same data partitions.

---

## Pipeline Overview

### YOLO-based End-to-End Pipeline

YOLO11 and YOLO26 are used as alternative segmentation front-ends within the complete TTI detection pipeline.

```text
Input Frame
    ↓
YOLO11 / YOLO26
    ↓
Tool and TTI-related Detections
    ↓
Tool–Tissue Candidate Pairs
    ↓
Depth Anything V2
    ↓
5-Channel ROI
(Image + Depth + Union Mask)
    ↓
ROIClassifier
    ↓
TTI Prediction
```

The same Depth Anything V2 model and ROIClassifier are used for both architectures.

### U-Net

A multiclass U-Net was also developed as an alternative semantic-segmentation approach for surgical tools and TTI regions.

```text
Input Frame
    ↓
Multiclass U-Net
    ↓
Tool / TTI Masks
    ↓
Depth Anything V2
    ↓
Depth-Based Candidate Contact Region
```

Since the U-Net showed lower segmentation performance than YOLO26 in the initial experiments, it was not integrated into the baseline end-to-end pipeline. The final controlled comparison therefore focuses on **YOLO11 and YOLO26**.

---

## Repository Structure

```text
MR_project/
│
├── Dataset/
│   ├── create_dataset.py            # Converts original data to YOLO and U-Net formats
│   └── dataset_summary.py           # Dataset statistics and consistency checks
│
├── checkpoints/                     # Trained U-Net checkpoints
│   ├── unet_multiclass_best.pth
│   ├── unet_multiclass_best_f1.pth
│   └── unet_multiclass_weighted_best_f1.pth
│
├── comparison/                      # End-to-end YOLO11 vs YOLO26 experiments
│   ├── 01_initial_comparison/       # Preliminary comparison
│   └── 02_final_comparison/         # Controlled comparison under identical settings
│
├── configs/
│   └── data_tools_yolo26.yaml       # YOLO dataset configuration and class mapping
│
├── outputs/                         # Generated experimental results
│   ├── comparison/
│   ├── unet/
│   ├── yolo11/
│   └── yolo26/
│
├── pdf/                             # Project material and reference documents
│
├── Baseline+Yolo11.ipynb            # End-to-end TTI pipeline with YOLO11
├── Baseline+Yolo26.ipynb            # End-to-end TTI pipeline with YOLO26
├── ROImodel.pt                      # ROIClassifier weights
├── labels.ipynb                     # Merges tool and TTI labels for YOLO
├── program(yolo+unet).ipynb         # YOLO, U-Net and depth-processing experiments
├── train_yolo11.ipynb               # YOLO11 training for the controlled comparison
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Models

### YOLO11 and YOLO26

YOLO11 and YOLO26 are used for instance segmentation and as alternative front-ends for the end-to-end TTI detection pipeline.

For the final controlled comparison, both models were trained using the same experimental settings:

- input resolution: **640 × 640**
- batch size: **16**
- maximum epochs: **150**
- early-stopping patience: **100**

### Multiclass U-Net

The U-Net provides an alternative dense semantic-segmentation approach. It produces multiclass masks that can be converted into tool and TTI masks.

Because its initial segmentation results were lower than those obtained with YOLO26, it was retained as an auxiliary experiment rather than being integrated into the final end-to-end comparison.

### Depth Anything V2

Depth Anything V2 is used for monocular **relative-depth estimation** and provides additional spatial information for contact-region estimation and ROI construction.

### ROIClassifier

The ResNet-18-based ROIClassifier performs downstream TTI classification using candidate tool–tissue ROIs generated from image, depth, and segmentation information.

---

## Final YOLO11 vs YOLO26 Comparison

An initial comparison was first performed using the provided YOLO11 baseline and the YOLO26 model trained during the project.

A final **controlled comparison** was then performed by training YOLO11 and YOLO26 under identical conditions and evaluating them using the same test set, Depth Anything V2 model, ROIClassifier, and frame-level evaluation protocol.

| Metric | YOLO11 | YOLO26 |
|---|---:|---:|
| Accuracy | **0.802** | 0.681 |
| Precision | 0.896 | **0.938** |
| Recall | **0.846** | 0.636 |
| F1-score | **0.871** | 0.758 |
| Specificity | 0.641 | **0.846** |

YOLO11 achieved the best overall trade-off, with higher accuracy, recall, and F1-score. YOLO26 achieved higher precision and specificity but produced substantially more false negatives.

---

## Authors

- Letizia Sorriento
- Federico De Angelis
- Claudia Julia Istoc
- Martina Leggiero

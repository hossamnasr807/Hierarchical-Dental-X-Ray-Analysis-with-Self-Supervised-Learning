# Hierarchical Dental X-Ray Analysis with Self-Supervised Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![LightlySSL](https://img.shields.io/badge/SSL-Lightly-orange)

A state-of-the-art deep learning pipeline for analyzing panoramic dental X-rays (OPGs). This project utilizes a **hierarchical "divide-and-conquer" strategy** combined with **Self-Supervised Learning (SimCLR)** to achieve high-precision quadrant detection and tooth enumeration, even with limited labeled data.

Designed for the **DENTEX Challenge** dataset.

![System Architecture](docs/pipeline_diagram.png)

## 🧠 Core Features

* **Self-Supervised Pre-training (SimCLR):** Uses unlabeled X-rays to train the backbone feature extractor before supervised fine-tuning. This improves robustness to artifacts and implants.
* **Hierarchical Detection Pipeline:**
    * **Stage 1 (Global):** Detects the 4 quadrants of the jaw to establish spatial context.
    * **Stage 2 (Local):** Crops each quadrant and detects individual teeth (FDI Notation 11-48) with high resolution.
* **Model-Assisted Dataset Generation:** Includes an intelligent script that auto-generates the Stage 2 training set by mapping global coordinates to local crops with adaptive padding.
* **Coordinate Mathematics:** Robust logic to map bounding boxes between global (full X-ray) and local (tooth crop) coordinate systems.

---

## 🏗️ Architecture Overview

The system operates in three distinct phases:

| Phase | Task | Model Architecture | Input Resolution |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **Feature Learning** | **SimCLR (ResNet/CSP-Darknet)** | 512x512 |
| **Phase 1** | **Quadrant Detection** | **YOLOv8n (Nano)** | 640x640 |
| **Phase 2** | **Teeth Enumeration** | **YOLOv8s (Small)** | 640x640 (Cropped) |

### Why Hierarchical?
Detecting 32 small, dense objects (teeth) on a massive 2000x1000 pixel image is difficult for single-shot detectors. By first locating the quadrants, we can crop the image, effectively **quadrupling the pixel density** for the tooth detector.

---

## 📦 Installation

This project requires PyTorch, Ultralytics YOLO, and Lightly (for SSL).

```bash
# Clone the repository
git clone https://github.com/hossamnasr807/Hierarchical-Dental-X-Ray-Analysis-with-Self-Supervised-Learning.git
cd Hierarchical-Dental-X-Ray-Analysis-with-Self-Supervised-Learning

# Install dependencies
pip install torch torchvision
pip install ultralytics
pip install lightly
pip install opencv-python matplotlib tqdm pyyaml

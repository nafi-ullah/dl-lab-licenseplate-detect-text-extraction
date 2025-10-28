# 🚗 Automatic Number Plate Recognition (ANPR)
### Deep Learning Project using PyTorch and EasyOCR

This project implements an **Automatic Number Plate Recognition (ANPR)** system that detects and recognizes vehicle license plates from **video feeds or images**.  
It uses a **Default YOLOv8 Model** for vehicle detection, **PyTorch-based segmentation model** for license plate localization and **EasyOCR** for optical character recognition (OCR) to extract text.

---

## 📘 Overview
Automatic Number Plate Recognition (ANPR) is a computer vision technique used for identifying vehicles by detecting and reading their license plates.  
This project demonstrates a complete ANPR pipeline that:
1. Takes in video or image input.
2. Segments the license plate region using a trained deep learning model.
3. Extracts and preprocesses the plate area.
4. Recognizes alphanumeric text using EasyOCR.

---

## 🧠 Project Architecture

### 🔹 Model Components
- **Segmentation Model (`license_plate_detector.pt`)**
  - Trained in PyTorch to generate a segmentation mask for the license plate region.
  - The mask is post-processed to obtain the bounding area for the plate.
- **OCR Engine**
  - EasyOCR extracts alphanumeric text from the segmented plate area.

### 🔹 Workflow
1. **Input:** Image or video frame.
2. **Segmentation:** Model predicts mask of license plate.
3. **Post-processing:** Extracts plate ROI from mask.
4. **OCR:** EasyOCR reads plate characters.
5. **Output:** Display or store recognized plate number with coordinates.


---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/nafi-ullah/dl-lab-licenseplate-detect-text-extraction.git
cd dl-lab-licenseplate-detect-text-extraction
```

### 2️⃣ Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### ▶️ Run Inference
```bash
#To process a video or image for license plate detection and text extraction:
python main.py --input path/to/video_or_image

#Use the visualization script to view segmentation masks and OCR outputs:
python visualize.py
```
---


## 📊 Dataset

| Property | Details |
|-----------|----------|
| **Total Images** | ~2300 annotated vehicle images |
| **Annotation Type** | Segmentation masks for license plates |
| **Preprocessing** | Resize, normalization, augmentation (rotation, brightness, noise) |
| **Dataset Split** | 80% train, 10% validation, 10% test |

---

## 🧾 Project Structure
```bash
dl-lab-licenseplate-detect-text-extraction/
│
├── main.py                   # Main pipeline for segmentation + OCR
├── util.py                   # Utility functions
├── visualize.py              # Visualization and output rendering
├── requirements.txt          # Dependencies
├── license_plate_detector.pt # Trained segmentation model
├── yolov8n.pt                # Pretrained reference model (if used)
├── license_plates_results.csv# Output results
└── README.md                 # Project documentation

```
---

## 🧠 Training Results

| **Metric**              | **Training** | **Validation** |
|--------------------------|--------------|----------------|
| **Box Loss**             | 0.60         | 0.75           |
| **Segmentation Loss**    | 0.85         | 0.95           |
| **Classification Loss**  | 0.30         | 0.35           |
| **mAP@50**               | 0.97         | 0.95           |
| **Precision**            | 0.95         | 0.90           |
| **Recall**               | 0.93         | 0.87           |

---

## 📈 Results & Performance

### 🔢 Quantitative Metrics

| **Metric** | **Value** | **Description** |
|-------------|------------|-----------------|
| **mAP @ IoU=0.5** | 0.88 | Mean Average Precision at 50% overlap |
| **mAP @ IoU=0.5:0.95** | 0.72 | Averaged across multiple IoU thresholds |
| **Precision** | 0.90 | Proportion of correct positive predictions |
| **Recall** | 0.87 | Proportion of actual plates detected |
| **F1-Score** | 0.885 | Harmonic mean of precision and recall |
| **OCR Character Accuracy** | 0.85 | Percentage of correctly recognized characters |
| **OCR Plate Accuracy** | 0.68 | Percentage of perfectly recognized license plates |
| **Inference Time (GPU)** | 40–50 ms | Average time per frame during inference |
| **FPS (GPU)** | 20–25 | Frames processed per second in real time |

---

### 🏆 Summary

- The model achieves **high accuracy** in both segmentation and OCR recognition.
- Maintains **real-time performance** (20–25 FPS) on GPU inference.
- Balanced trade-off between **precision and recall**, ensuring robust plate detection and readable text extraction.

---

## 👩‍💻 Author

**Nafi Ullah Shafin & Sadia Farzana Jessi**  
*Neural Network & Deep Learning Laboratory Project*  
**Software Engineering (IICT), SUST**
---

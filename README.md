# CO2Wounds-V2: Digital Twin in Healthcare for Chronic Wound Management

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A Deep Learning Based System for Chronic Wound Detection, Classification, and Remedy Recommendation in Leprosy Patients**

## 📌 Project Overview

**CO2Wounds-V2** addresses the challenge of diagnosing and managing chronic wounds in leprosy patients using artificial intelligence. Leveraging deep learning models like **VGG-19**, **InceptionV3**, and **UNet++**, the project provides an end-to-end solution from wound image input to remedy recommendation. A web-based application built with **Streamlit** enhances accessibility for real-time clinical use.

---

## 🎯 Objectives

- ✅ Classify chronic wound types using transfer learning (VGG-19, InceptionV3).
- ✅ Segment affected regions precisely using UNet++.
- ✅ Extract statistical and texture features like **Mean**, **Median**, **Variance**, and **GLCM**.
- ✅ Recommend suitable treatment options via a web application.
- ✅ Improve consistency, speed, and accuracy of wound diagnosis in leprosy care.

---

## 🧠 Technologies & Algorithms

| Component           | Technology/Library         |
|---------------------|----------------------------|
| Language            | Python                     |
| Deep Learning       | VGG-19, InceptionV3, UNET++|
| Feature Extraction  | GLCM, Mean, Variance       |
| Web App             | Streamlit                  |
| Data Management     | SQLite                     |
| Image Processing    | OpenCV, Matplotlib, PIL    |
| ML Metrics          | Accuracy, Precision, Recall, F1-Score |

---

## 📁 Dataset

- **Name**: [CO2Wounds-V2 Extended Chronic Wounds Dataset](https://ieee-dataport.org/open-access/co2wounds-v2-extended-chronic-wounds-dataset-leprosy-patients-segmentation-and-detection)
- **Size**: 2,940 images | 104MB
- **Categories**:
  - Abrasions
  - Bruises
  - Burns
  - Cuts
  - Diabetic Wounds
  - Lacerations
  - Normal
  - Pressure Wounds
  - Surgical Wounds
  - Venous Wounds

---
## 🖼 Sample Interface Screens
-  ✅ Login/Registration System with SQLite

-  📤 Upload Image UI with real-time prediction

-  📈 Display of diagnostic results (wound type, metrics, remedies)


## 📊 Performance Metrics

We evaluated our models on standard machine learning metrics to ensure clinical relevance and accuracy.

### 🔍 Classification Results

| Metric       | VGG-19 | InceptionV3 |
|--------------|--------|-------------|
| Accuracy     | 97%    | 95%         |
| Precision    | 87%    | 85%         |
| Recall       | 93%    | 92%         |
| F1-Score     | 90.06% | 88.83%      |
| Error Rate   | 3%     | 5%          |

> Metrics are computed using test data from the CO2Wounds-V2 dataset containing 10 wound classes.
---

## 🏥 Applications

- 🔬 Clinical support for leprosy wound diagnosis  
- 🌐 Remote wound monitoring and treatment suggestion  
- 📱 AI-based mobile integration for rural healthcare  

---

## ⚠️ Limitations

- ⚙️ Requires GPU for best inference speed  
- 🔍 Lack of patient metadata (age, healing history) may limit personalization  
- 📉 Generalizability across different skin tones and lighting needs enhancement  

---

## 🚀 Future Work

- 📦 Add metadata like healing stage, demographics  
- 🧬 Integrate real-time image capture via mobile  
- 🌐 Expand platform to include diabetic and pressure ulcers  
- 🧠 Deploy on cloud for remote clinics  

---


## 🧪 Methodology
---
```mermaid
graph TD
A[Input Image Upload] --> B[Preprocessing Resize, Grayscale, Filter]
B --> C[Feature Extraction Mean, Median, GLCM]
C --> D[Segmentation - UNET++]
D --> E[Classification - VGG19 & InceptionV3]
E --> F[Prediction: Wound Type & Severity]
F --> G[Remedy Suggestion]
G --> H[Web Display via Streamlit]



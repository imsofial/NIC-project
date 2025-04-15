# Brain Tumor Classification Using Spiking Neural Networks (SNNs)

## 📌 Project Description

This project aims to classify brain tumors from MRI scans using Spiking Neural Networks (SNNs). We explore the effectiveness of SNNs compared to traditional Convolutional Neural Networks (CNNs) for image classification. The goal is to evaluate whether SNNs provide advantages in terms of accuracy and computational efficiency.

## 🔧 Technologies & Tools

- Python
- Jupyter Notebook
- TensorFlow (Keras) / PyTorch (for CNNs)
- Brian2 / BindsNET (for SNNs)
- OpenCV / NumPy / Matplotlib/ Sklearn (for image preprocessing)

## 📂 Repository Structure

```
📁 brain-tumor-classification
│── 📁 brain-tumor-mri-dataset      # MRI dataset
│── 📁 models                       # Trained models (CNN and SNN)
│── results.ipynb                   # Visualizations and evaluation results
│── readme.md                       # This file
```

## 📊 Dataset

Dataset used: [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

Tumor categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

## 🎯 Workflow

✔️ Image preprocessing (normalization, noise reduction)  
✔️ Encoding images into spike trains  
✔️ Training an SNN using STDP learning rule  
✔️ Comparing SNN and CNN using evaluation metrics (accuracy, precision, recall, F1-score)  
✔️ Analyzing results and visualization

---

## 🛠 Methodology

### 🧬 Preprocessing
- All MRI images are resized (e.g., 240x240).
- Pixel values normalized.
- Optional augmentation for CNN (rotation, zoom, flip).
- For SNN: Encoded images into **spike trains**.

### 🧠 SNN Implementation
- Implemented using **Brian2 / BindsNET**.
- Learning via **STDP (Spike-Timing Dependent Plasticity)**.
- Evaluation with test split using temporal input encodings.

### 🤖 CNN Implementation
- Architecture:
    - 3x Convolutional + MaxPooling + BatchNorm
    - 3x Dense layers with Dropout
    - Final Softmax for 4-class output
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 10

### 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🧪 Timeline & Responsibilities

| Week | Task                                               | Assigned To               |
|------|----------------------------------------------------|----------------------------|
| 1-2  | Research SNN theory and image encodings            | Ekaterina, Sofia           |
| 3-4  | SNN model implementation                           | Ekaterina                  |
| 3-4  | CNN architecture and setup                         | Yasmina                    |
| 5-6  | Training both models & evaluations                 | All                        |
| 7-8  | Visualization, reporting, and final presentation   | All                        |


### 📌 Week 1–2: Research & Preprocessing
- **Goal:** Understand SNNs and data preprocessing
- **Tasks:**
    - 📚 *Ekaterina*: Researched spike encoding strategies and initial data cleaning
    - 🧠 *Sofia*: Studied SNN architectures and the STDP learning rule
    - 🗂️ *Yasmina*: Prepared and structured the dataset for training

---

### 📌 Week 3–4: SNN Development
- **Goal:** Build and train the SNN model
- **Tasks:**
    - 👩‍💻 *Ekaterina*: Implemented the SNN using `Brian2`/`BindsNET`
    - 🔧 *Sofia*: Designed and tuned the learning process for the SNN
    - 🤖 *Yasmina*: Set up the CNN baseline model for comparison

---

### 📌 Week 5–6: Training & Evaluation
- **Goal:** Train and compare both models
- **Tasks:**
    - 🧪 *Ekaterina*: Finalized training of the SNN and logged results
    - 🔍 *Sofia*: Trained CNN and benchmarked against SNN
    - 📊 *Yasmina*: Aggregated all metrics (accuracy, F1, confusion matrix)

---

### 📌 Week 7–8: Analysis & Reporting
- **Goal:** Visualize, report, and present findings
- **Tasks:**
    - 📈 *Ekaterina*: Created result visualizations and performance graphs
    - 📝 *Sofia*: Wrote final analysis and conclusions
    - 🎤 *Yasmina*: Compiled final report and built presentation slides

---

## 🔍 Analysis

| Model | Accuracy | F1 Score | Precision | Recall | Energy-Efficiency |
|-------|----------|----------|-----------|--------|-------------------|
| CNN   | ✅ 96.5% | ✅ 96.4% | ✅ 96.4%  | ✅ 96.5% | ❌ High GPU usage |
| SNN   | ❌ 77.9% | ❌ 78.1% | ❌ 79.0%  | ❌ 77.8% | ✅ Bio-inspired, low power |

- CNNs are **highly accurate**, but require more computation.
- SNNs are **less accurate** but **energy-efficient** and better suited for edge devices or real-time applications.

---

## 📚 References

- 🧠 [Kaggle Dataset – Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- 🧬 [Brian2 Documentation](https://brian2.readthedocs.io/en/stable/)
- ⚡ [BindsNET Library](https://bindsnet-docs.readthedocs.io/en/stable/)
- 🧠 [Spiking Neural Networks (SNN) Paper](https://arxiv.org/abs/1808.02564)

---

## 🧑‍💻 Contributors

- **Ekaterina Akimenko** 
- **Sofia Goryunova**
- **Yasmina Mamadalieva** 

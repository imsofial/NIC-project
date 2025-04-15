# Brain Tumor Classification Using Spiking Neural Networks (SNNs)

## 📌 Project Description

This project aims to classify brain tumors from MRI scans using Spiking Neural Networks (SNNs). We explore the effectiveness of SNNs compared to traditional Convolutional Neural Networks (CNNs) for image classification. The goal is to evaluate whether SNNs provide advantages in terms of accuracy and computational efficiency.

## 🔧 Technologies & Tools

- Python
- Jupyter Notebook
- PyTorch + snntorch (for SNN implementation)
- TensorFlow / Keras (for CNN baseline)
- scikit-learn (for evaluation metrics)
- OpenCV, NumPy, Matplotlib (for preprocessing and visualization)

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

--

## 🎯 Workflow

✔️ Image preprocessing (grayscale, resizing, normalization)  
✔️ CNN: classic convolutional layers + data augmentation
✔️ SNN: LIF neurons using surrogate gradient learning
✔️ Rate-based spike encoding
✔️ Comparing SNN and CNN using evaluation metrics (accuracy, precision, recall, F1-score)  
✔️ Analyzing results and visualization

---

## 🛠 Methodology

### 🧬 Preprocessing
- All images resized to 240x240 (CNN) and 64x64 (SNN)
- Normalized pixel values to [-1, 1]
- CNN: data augmentation (rotation, flips, zoom)
- SNN: grayscale + spike encoding over 10 timesteps

### 🧠 SNN Implementation
- Implemented using snntorch (based on PyTorch)
- Neuron type: Leaky Integrate-and-Fire (LIF)
- Learning method: backpropagation with surrogate gradients
- Temporal encoding via rate-based spike generation
- Final classification via fully connected layer

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

---

### 📌 Week 1–2: Research & Preprocessing
- **Goal:** Study SNNs and prepare dataset
- **Tasks:**
  - 📚 *Ekaterina*: Researched rate-based spike encoding and SNN neuron models (LIF)
  - 🧠 *Sofia*: Investigated surrogate gradient learning and neuron behavior in `snntorch`
  - 🗂️ *Yasmina*: Organized the dataset into train/test folders and implemented basic preprocessing pipelines

---

### 📌 Week 3–4: SNN & CNN Implementation
- **Goal:** Implement SNN and CNN models
- **Tasks:**
  - 👩‍💻 *Ekaterina*: Built the SNN architecture using `snntorch` with surrogate gradients and LIF neurons
  - 🔧 *Sofia*: Integrated spike encoding pipeline and helped implement dropout, normalization, and pooling
  - 🤖 *Yasmina*: Implemented the CNN baseline using Keras with augmentation and softmax classifier

---

### 📌 Week 5–6: Model Training & Evaluation
- **Goal:** Train both models and compare results
- **Tasks:**
  - 🧪 *Ekaterina*: Trained the SNN model, monitored training dynamics, and optimized accuracy
  - 🔍 *Sofia*: Supported training and helped evaluate both models using precision/recall/F1
  - 📊 *Yasmina*: Trained the CNN, collected confusion matrices, and evaluated both models on full test set

---

### 📌 Week 7–8: Final Analysis & Reporting
- **Goal:** Evaluate models, visualize metrics, and write final report
- **Tasks:**
  - 📈 *Ekaterina*: Produced result plots and cross-model performance comparisons
  - 📝 *Sofia*: Wrote the results, analysis, and key insights sections
  - 🎤 *Yasmina*: Prepared final report, README, and presentation slides for submission

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
- 🧬 [snntorch Documentation](https://snntorch.readthedocs.io/)
- 🧠 [Spiking Neural Networks (SNN) Paper](https://arxiv.org/abs/1808.02564)

---

## 🧑‍💻 Contributors

- **Ekaterina Akimenko** 
- **Sofia Goryunova**
- **Yasmina Mamadalieva** 

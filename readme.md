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

## 🧑‍💻 Contributors

- **Ekaterina Akimenko** – Research on encoding techniques & preprocessing
- **Sofia Goryunova** – SNN architecture & STDP implementation
- **Yasmina Mamadalieva** – Dataset preparation & CNN development

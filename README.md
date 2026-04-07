# Malaria Cell Image Classification using MobileNetV2

## Overview

This project presents a deep learning-based system for detecting malaria infection from microscopic blood smear images. It uses transfer learning with MobileNetV2 to classify cells as parasitized or uninfected. The system is designed to support fast and accurate diagnosis, especially in low-resource healthcare settings.

---

## Objectives

- Classify blood smear images into parasitized and uninfected categories  
- Leverage transfer learning to improve model performance  
- Improve generalization using data augmentation techniques  
- Evaluate model performance using multiple metrics  
- Generate visual insights for better understanding of predictions  

---

## System Architecture

The system follows a structured pipeline consisting of the following stages:

1. Dataset Loading and Preprocessing  
2. Data Augmentation  
3. Feature Extraction using MobileNetV2  
4. Model Training (Classification Head)  
5. Fine-Tuning of Deeper Layers  
6. Model Evaluation  
7. Visualization of Results  

---

## Model Details

- Base Model: MobileNetV2 (pretrained on ImageNet)  
- Classification Head:
  - Global Average Pooling  
  - Dense Layers with Batch Normalization  
  - Dropout for regularization  
  - Sigmoid output layer for binary classification  

---

## Training Strategy

- Stage 1: Freeze base model and train top layers  
- Stage 2: Unfreeze deeper layers and fine-tune  

### Techniques Used

- Data augmentation (flipping, brightness, contrast)  
- EarlyStopping to prevent overfitting  
- ModelCheckpoint to save best model  
- ReduceLROnPlateau for learning rate adjustment  

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- AUC Score  
- Confusion Matrix  
- ROC Curve  

---

## Outputs

- Training performance graphs  
- Confusion matrix  
- ROC curve  
- Sample predictions  

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## Objective

The goal of this project is to build a scalable and reliable AI-based solution that can assist healthcare professionals in early malaria detection and reduce manual diagnostic effort.

---

## Future Improvements

- Deploy as a web application using Flask or Streamlit  
- Convert model to TensorFlow Lite for mobile deployment  
- Integrate with cloud platforms such as AWS or GCP  
- Add explainability techniques like Grad-CAM  

---

## License

This project is open-source and available under the MIT License.

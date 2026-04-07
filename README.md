# Malaria Cell Image Classification using MobileNetV2

## Overview

This project presents a deep learning-based system for detecting malaria infection from microscopic blood smear images. It leverages transfer learning with MobileNetV2 to classify cells as parasitized or uninfected. 

The system is enhanced with an interactive Streamlit web application and Grad-CAM visualization to provide both predictions and visual explanations, making it suitable for real-world healthcare assistance and AI-based diagnostics.

---

## Objectives

- Classify blood smear images into parasitized and uninfected categories  
- Leverage transfer learning to improve model performance  
- Improve generalization using data augmentation techniques  
- Evaluate model performance using multiple metrics  
- Provide an interactive web interface for real-time predictions  
- Generate visual explanations using Grad-CAM  

---

## System Architecture

The system follows a structured pipeline consisting of the following stages:

1. Dataset Loading and Preprocessing  
2. Data Augmentation  
3. Feature Extraction using MobileNetV2  
4. Model Training (Classification Head)  
5. Fine-Tuning of Deeper Layers  
6. Model Evaluation  
7. Streamlit Web App Deployment  
8. Grad-CAM Visualization for Explainability  

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

- Accuracy: ~95%  
- Precision and Recall: Balanced performance  
- AUC Score: ~0.98  
- Confusion Matrix  
- ROC Curve  

---

## Outputs

- Training performance graphs  
- Confusion matrix  
- ROC curve  
- Sample predictions  
- Interactive Streamlit web application  
- Grad-CAM heatmaps for model explainability  

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Streamlit  
- OpenCV  

---

## Objective

The goal of this project is to build a scalable and reliable AI-based solution that can assist healthcare professionals in early malaria detection, reduce manual diagnostic effort, and provide explainable AI outputs for better trust and interpretability.

---

## Future Improvements

- Deploy application on cloud platforms for public access  
- Convert model to TensorFlow Lite for mobile deployment  
- Improve Grad-CAM with advanced techniques (Grad-CAM++)  
- Add real-time camera input support  
- Optimize model for faster inference  

---

## License

This project is open-source and available under the MIT License.

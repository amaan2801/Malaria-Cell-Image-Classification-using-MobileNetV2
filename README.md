# 🧬 Malaria Cell Image Classification using MobileNetV2

A high-performance deep learning system for automated detection of malaria infection from microscopic blood smear images. This project leverages Transfer Learning with TensorFlow and MobileNetV2 to classify cells as **Parasitized** or **Uninfected** with high accuracy.

---

## 🚀 Features

* 📦 Automated dataset loading using TensorFlow Datasets (malaria dataset)
* 🧠 Transfer Learning with MobileNetV2 (ImageNet pretrained)
* ⚙️ Two-stage training strategy:

  * Stage 1: Train classification head
  * Stage 2: Fine-tune deeper layers
* 🔁 Data augmentation:

  * Random flips
  * Brightness & contrast adjustments
* 📊 Evaluation metrics:

  * Accuracy, Precision, Recall, AUC
  * Confusion Matrix & ROC Curve
* 📈 Visualization generation:

  * Training history graphs
  * Predictions & sample outputs
* 💾 Model saving:

  * `best_model.h5`
  * `malaria_detector_final.h5`
* ⚡ GPU compatible (Google Colab / CUDA)

---

## 📂 Project Structure

```
Malaria-Diagnosis/
│
├── malaria_diagnosis.py
├── models/
│   ├── best_model.h5
│   └── malaria_detector_final.h5
│
├── visualizations/
│   ├── samples.png
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── predictions.png
│
└── README.md
```

---

## 🛠️ Requirements

Install dependencies:

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib seaborn scikit-learn
```

### Recommended:

* Python 3.8+
* GPU (Google Colab / CUDA)

---

## 🧪 How to Run

```bash
python malaria_diagnosis.py
```

### The script will automatically:

* Load & preprocess dataset
* Split into train/validation/test sets
* Train MobileNetV2 model
* Fine-tune deeper layers
* Evaluate performance
* Save models
* Generate visualizations

---

## 🧠 Model Architecture

* Base Model: MobileNetV2 (ImageNet pretrained)
* Custom Head:

  * Global Average Pooling
  * Dense (256) + BatchNorm + Dropout (0.5)
  * Dense (128) + BatchNorm + Dropout (0.3)
  * Dense (1, Sigmoid)

---

## 📊 Training Strategy

* Stage 1: Freeze base model, train top layers
* Stage 2: Unfreeze last ~50 layers and fine-tune

### Callbacks:

* ModelCheckpoint
* EarlyStopping
* ReduceLROnPlateau

---

## 📈 Outputs

Generated in `/visualizations`:

* Training history graphs
* Confusion matrix
* ROC curve
* Sample predictions

---

## 🔍 Inference Example

```python
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('models/malaria_detector_final.h5')

def predict(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred < 0.5:
        return f"Parasitized ({(1 - pred) * 100:.1f}% confidence)"
    else:
        return f"Uninfected ({pred * 100:.1f}% confidence)"
```

---

## 🎯 Objective

To build a scalable and reliable AI system for malaria detection that can assist healthcare professionals in early diagnosis, especially in low-resource settings.

---

## ⭐ Future Improvements

* 🌐 Deploy as a web app (Flask / Streamlit)
* 📱 Convert to TensorFlow Lite for mobile use
* ☁️ Cloud deployment (AWS / GCP)
* 🔍 Model explainability (Grad-CAM)

---

## 📚 Resources

* TensorFlow Documentation
* TensorFlow Datasets
* MobileNetV2 Research Paper

---

## 📜 License

This project is open-source and available under the MIT License.

Malaria Cell Image Classification using MobileNetV2

This project focuses on building a deep learning system to automatically detect malaria infection from microscopic blood smear images. It uses transfer learning with MobileNetV2 and TensorFlow to classify cells as either parasitized or uninfected. The aim is to create a reliable and efficient solution that can support early diagnosis, especially in areas with limited medical resources.

Overview

The model is built using a pretrained MobileNetV2 network, which helps in extracting meaningful features from images without training from scratch. A custom classification layer is added on top to perform binary classification.

The training process is divided into two stages. In the first stage, the base model remains frozen and only the top layers are trained. In the second stage, selected deeper layers are fine-tuned to improve the model’s performance and adaptability to the dataset.

To make the model more robust, data augmentation techniques such as flipping and brightness adjustments are applied. This helps the model generalize better to unseen data.

Features

The system automatically loads and preprocesses the malaria dataset using TensorFlow Datasets. It evaluates performance using key metrics such as accuracy, precision, recall, and AUC.

It also generates useful visualizations including training performance graphs, confusion matrix, ROC curve, and prediction samples. These outputs help in understanding how well the model is performing.

The trained models are saved and can be reused for inference. The entire implementation supports GPU acceleration, making it suitable for faster training on platforms like Google Colab.

Model Details

The architecture is based on MobileNetV2 pretrained on ImageNet. A lightweight classification head is added, consisting of fully connected layers with batch normalization and dropout to prevent overfitting. The final output layer uses a sigmoid activation function for binary classification.

Workflow

The complete pipeline includes dataset loading, preprocessing, train-test splitting, model training, fine-tuning, evaluation, and result visualization. All steps are automated within a single script, making the project easy to run and reproduce.

Objective

The primary goal of this project is to develop a scalable AI-based solution that can assist healthcare professionals in detecting malaria quickly and accurately, reducing dependency on manual analysis.

Future Scope

The project can be extended by deploying it as a web application, converting it for mobile devices using TensorFlow Lite, integrating with cloud platforms, and adding model explainability techniques such as Grad-CAM to better interpret predictions.

License

This project is open-source and available under the MIT License.

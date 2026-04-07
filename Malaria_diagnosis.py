import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os

# ----------------------------
# Config
# ----------------------------
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 5

# ----------------------------
# Load Dataset
# ----------------------------
print("Loading dataset...")

dataset, info = tfds.load("malaria", with_info=True, as_supervised=True)

train_data = dataset["train"]

# Split manually (80/10/10)
train_size = int(0.8 * info.splits["train"].num_examples)
val_size = int(0.1 * info.splits["train"].num_examples)

train_ds = train_data.take(train_size)
val_ds = train_data.skip(train_size).take(val_size)
test_ds = train_data.skip(train_size + val_size)

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    return image, label

train_ds = train_ds.map(preprocess).map(augment).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(BATCH_SIZE)

# ----------------------------
# Model
# ----------------------------
print("Building model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ----------------------------
# Callbacks
# ----------------------------
os.makedirs("models", exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/best_model.h5",
        monitor="val_loss",
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.3,
        patience=2
    )
]

# ----------------------------
# Training - Stage 1
# ----------------------------
print("\nTraining Stage 1...")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks
)

# ----------------------------
# Fine-tuning
# ----------------------------
print("\nFine-tuning...")

base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks
)

# ----------------------------
# Evaluation
# ----------------------------
print("\nEvaluating...")

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(preds.flatten())

y_pred_labels = [1 if p > 0.5 else 0 for p in y_pred]

print("\nClassification Report:")
print(classification_report(y_true, y_pred_labels))

print("\nAUC Score:", roc_auc_score(y_true, y_pred))

# ----------------------------
# Save Final Model
# ----------------------------
model.save("models/malaria_detector_final.h5")

print("\nModel saved successfully!")

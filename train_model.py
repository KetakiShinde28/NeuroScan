import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# ✅ Define Dataset Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "Training")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "Testing")

# ✅ Check If Dataset Exists
if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
    raise FileNotFoundError("🚨 Dataset folder not found! Check the paths.")

IMG_SIZE = 128
BATCH_SIZE = 32
NUM_CLASSES = 4  # Glioma, Meningioma, Pituitary, No Tumor

# ✅ Data Augmentation (Balanced)
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    brightness_range=[0.85, 1.15],  
    horizontal_flip=True,
    shear_range=0.15,
    validation_split=0.2  
)

# ✅ Load Training and Validation Data
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ✅ Load Pretrained VGG16 Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Initially Freeze All Layers

# ✅ Unfreeze Last Few Layers for Fine-Tuning
for layer in base_model.layers[-6:]:
    layer.trainable = True

# ✅ Define Improved Model Architecture
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation="relu", kernel_regularizer=l2(0.005)),  # Increased regularization
    BatchNormalization(),
    Dropout(0.6),  # Increased Dropout to prevent overfitting
    Dense(NUM_CLASSES, activation="softmax")
])

# ✅ Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

# ✅ Compile Model
optimizer = Adam(learning_rate=0.0001)  
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ Implement Early Stopping & LR Scheduling
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ✅ Train the Model
history = model.fit(train_generator, 
                    validation_data=val_generator, 
                    epochs=25,  
                    callbacks=[early_stopping, lr_scheduler])

# ✅ Save the Trained Model
model.save("../backend/brain_tumor_model.keras")
print("✅ Model training complete and saved as 'brain_tumor_model.keras'")

# ✅ Load Test Data
test_generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ✅ Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# ✅ Generate Predictions
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes  
class_labels = list(test_generator.class_indices.keys())

# ✅ Print Classification Report
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_labels))

# ✅ Plot Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d",
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

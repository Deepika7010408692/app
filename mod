import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# Load and preprocess data
def load_images(folder_path):
    images, labels = [], []
    for label in ["PNEUMONIA", "NORMAL"]:
        path = os.path.join(folder_path, label)
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                img = cv2.resize(img, (128, 128)) / 255.0  # Normalizing the images
                images.append(img)
                labels.append(1 if label == "PNEUMONIA" else 0)
    return np.array(images), np.array(labels)

# Paths to data directories
train_images, train_labels = load_images('./train')
val_images, val_labels = load_images('./val')
test_images, test_labels = load_images('./test')

# Define CNN model
def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the CNN model
cnn_model = build_cnn_model()
cnn_model.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))

# Save the model
cnn_model.save('cnn_model.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = cnn_model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy:.2f}")

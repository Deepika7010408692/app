import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import streamlit as st

# Load and preprocess data
def load_images(folder_path):
    images, labels = [], []
    for label in ["PNEUMONIA", "NORMAL"]:
        path = os.path.join(folder_path, label)
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                img = cv2.resize(img, (128, 128)) / 255.0  # Normalize to [0, 1]
                images.append(img)
                labels.append(1 if label == "PNEUMONIA" else 0)
    return np.array(images), np.array(labels)

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
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load pre-trained model or train a new one if not available
cnn_model = build_cnn_model()
# Optionally, load saved weights if available
# cnn_model.load_weights('chest_xray_cnn_model.h5')

# Streamlit app for pneumonia detection
st.title("Pneumonia Detection from Chest X-rays")

# Function to preprocess a single image
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction function
def predict_pneumonia(image):
    preprocessed_image = preprocess_image(image)
    prediction = cnn_model.predict(preprocessed_image)[0][0]
    result = "Pneumonia Detected" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return result, confidence

# Upload and predict
uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="RGB", caption="Uploaded Image", use_column_width=True)
    
    # Perform prediction
    result, confidence = predict_pneumonia(image)
    st.write(f"**Result:** {result}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

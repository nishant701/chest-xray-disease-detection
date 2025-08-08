import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

IMG_SIZE = 224
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_image(image_bytes):
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)
    img = clahe.apply(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def load_trained_model(model_path='model/chest_disease_model_final.h5'):
    return load_model(model_path)

def predict_disease(model, image_array, class_labels):
    prediction = model.predict(image_array)[0]
    class_index = np.argmax(prediction)
    return class_labels[class_index], float(prediction[class_index])

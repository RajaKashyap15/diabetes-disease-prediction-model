import sys
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "brain_tumor_cnn.h5")
print(f"Using model: {MODEL_PATH}")
model = load_model(MODEL_PATH)

class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# predict_image(img_path) - this function load image and predict tumor class
def predict_image(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]

    print(f"\nPredictions for {os.path.basename(img_path)}:")
    for cls, prob in zip(class_names, prediction):
        print(f"   • {cls:<12} → {prob*100:.2f}%")

    final_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    print(f"\nFinal Prediction: {final_class} ({confidence:.2f}% confidence)")

    plt.figure(figsize=(6,4))
    plt.bar(class_names, prediction*100, color="skyblue", edgecolor="black")
    plt.title(f"Prediction: {final_class} ({confidence:.2f}%)")
    plt.ylabel("Confidence (%)")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_image(sys.argv[1])

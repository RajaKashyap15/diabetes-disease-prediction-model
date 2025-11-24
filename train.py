import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(
    BASE_DIR,
    "D:\\NullDoc\\Brain_Tumor-Classifier\\Data\\BT-MRI Dataset\\BT-MRI Dataset\\Training"
)


test_dir = os.path.join(
    BASE_DIR,
    "D:\\NullDoc\\Brain_Tumor-Classifier\\Data\\BT-MRI Dataset\\BT-MRI Dataset\\Testing"
)



print(f"Training dir: {train_dir}")
print(f"Testing dir: {test_dir}")

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.20),
    tf.keras.layers.RandomContrast(0.20),
])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)


train_datagen = ImageDataGenerator(rescale=1/255.0)
test_datagen = ImageDataGenerator(rescale=1/255.0)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# main train script - build model, train and save weights
model = build_model(input_shape=(224,224,3), num_classes=train_gen.num_classes)

history = model.fit(
    train_gen,
    epochs=20,
    validation_data=test_gen
)

model.save("brain_tumor_vgg16.h5")
print("Model saved as brain_tumor_vgg16.h5")

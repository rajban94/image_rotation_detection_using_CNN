import os
import json
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
model = tf.keras.models.load_model('./Model/rotation_model.h5')

# Load the class labels
with open('./Model/classes.json', 'r') as f:
    classes = json.load(f)

# Set the image dimensions
image_height = 224
image_width = 224

# Function to predict the rotation angle of an image
def predict_rotation(image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(image_height, image_width))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Predict the rotation angle
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)
    predicted_angle = classes[predicted_class_index]

    return predicted_angle

# Set the path to the test image
test_image_path = './Model/test_3_270.png'

# Predict the rotation angle of the test image
predicted_angle = predict_rotation(test_image_path)

print('Predicted Rotation Angle:', predicted_angle)

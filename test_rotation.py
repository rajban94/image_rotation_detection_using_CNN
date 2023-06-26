############## Testing 1 ##########################

import os
import json
import numpy as np
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



############## Testing 2 ##########################

# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image

# # Load the trained model
# model = load_model("trained_model.h5")

# # Load and preprocess the test image
# def preprocess_image(image):
#     image = image.resize((224, 224))  # Resize the image to match the input size of the model
#     image = tf.keras.preprocessing.image.img_to_array(image)  # Convert image to numpy array
#     image = tf.keras.applications.mobilenet_v3.preprocess_input(image)  # Preprocess the image as required by MobileNetV3
#     return image

# def test_image(image_path):
#     # Load and preprocess the test image
#     image = Image.open(image_path)
#     image = preprocess_image(image)

#     # Reshape the image to match the model input shape
#     image = tf.reshape(image, (1, 224, 224, 3))

#     # Perform inference
#     predicted_angle = model.predict(image)[0][0]

#     return predicted_angle

# # Test the model on an unseen image
# test_image_path = "/content/test_inv_page1.jpg"
# predicted_angle = test_image(test_image_path)
# print("Predicted Angle:", predicted_angle)

import os
import json
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

# Define the CNN model
def create_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))  # Output layer with 4 classes (0, 90, 180, 270)
    return model

# Load and preprocess the dataset from images folder and JSON file
def load_dataset(images_folder, annotations_file):
    X = []
    y = []
    
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
        
    for entry in annotations['images']:
        image_path = entry['filename'] #os.path.join(images_folder, image_name)
        image = load_img(image_path, target_size=(image_height, image_width))
        image = img_to_array(image)
        X.append(image)
        y.append(entry['angle'])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# Set the path to your images folder and annotations JSON file
images_folder = '/content/rotated_dir_another'
annotations_file = '/content/image_rotation_data_another.json'

# Set the image dimensions
image_height = 224
image_width = 224

# Load and preprocess the dataset
X, y = load_dataset(images_folder, annotations_file)

# Convert labels to numeric values
label_mapping = {'0': 0, '90': 1, '180': 2, '270': 3}
y = np.array([label_mapping[label] for label in y])

# Split the dataset into training, validation, and testing sets
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Create the CNN model
model = create_model()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Save the model and necessary files for standalone testing
model.save('rotation_model.h5')  # Save the trained model
classes = list(label_mapping.keys())  # Define the class labels
with open('classes.json', 'w') as f:
    json.dump(classes, f)  # Save the class labels to a JSON file

# Evaluate the model on the testing set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
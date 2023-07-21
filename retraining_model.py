import os
import json
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

# Set the path to your images folder and annotations JSON file
images_folder = 'path/to/images/folder'
annotations_file = 'path/to/annotations.json'

# Set the image dimensions
image_height = 224
image_width = 224

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

# Load and preprocess the dataset
X, y = load_dataset(images_folder, annotations_file)

# Convert labels to numeric values
label_mapping = {'0': 0, '90': 1, '180': 2, '270': 3}
y = np.array([label_mapping[label] for label in y])

# Split the dataset into training, validation, and testing sets
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Load the saved model
saved_model_path = 'rotation_model.h5'
model = tf.keras.models.load_model(saved_model_path)

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Define a checkpoint to save the best model
checkpoint_path = 'best_model.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Continue training the model
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val), callbacks=[checkpoint])

# Save the retrained model
model.save('retrained_rotation_model.h5')

# Evaluate the model on the testing set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

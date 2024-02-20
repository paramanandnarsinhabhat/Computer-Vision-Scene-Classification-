# Import necessary libraries
import zipfile
import os
import pandas as pd
from PIL import Image
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define paths
zip_file_path = '/Users/paramanandbhat/Downloads/train-scene classification.zip'
extraction_directory = '/Users/paramanandbhat/Downloads/train_scene'
test_csv_path = '/Users/paramanandbhat/Downloads/test_hAjxzwh.csv'
images_dir = '/Users/paramanandbhat/Downloads/train_scene/train/'
output_csv_path = '/Users/paramanandbhat/Downloads/train_scene/output_predictions.csv'

# Function to create directories and unzip dataset
def setup_environment(zip_path, extract_dir):
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction completed.")

# Function to load and preprocess images
def preprocess_image(image_path, target_size=(128, 128)):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array

# Load dataset and preprocess images
def load_and_preprocess_data(csv_path, images_dir):
    data = pd.read_csv(csv_path)
    data['processed_image'] = data['image_name'].apply(lambda x: preprocess_image(os.path.join(images_dir, x)))
    return data

# Plotting function for data visualization
def plot_data(data, categories):
    fig, axes = plt.subplots(len(categories), 2, figsize=(10, 15))
    for i, category in enumerate(categories):
        category_data = data[data['label'] == i].head(2)
        for j, (_, row) in enumerate(category_data.iterrows()):
            axes[i, j].imshow(row['processed_image'], cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(category)
    plt.tight_layout()
    plt.show()

# Define and compile the model
def define_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(6, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main script starts here
setup_environment(zip_file_path, extraction_directory)
train_data = load_and_preprocess_data(os.path.join(extraction_directory, 'train.csv'), images_dir)
categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
plot_data(train_data, categories)

# Assuming model training and validation code goes here

# Load test data, preprocess, and predict
test_data = load_and_preprocess_data(test_csv_path, images_dir)
model = define_model()  # Load a pre-trained model if available
# model = load_model('path_to_your_saved_model.h5')
X_test = np.array(test_data['processed_image'].tolist())
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
test_data['label'] = predicted_labels
test_data[['image_name', 'label']].to_csv(output_csv_path, index=False)
print(f"Predictions saved to {output_csv_path}")

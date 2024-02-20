# Import necessary libraries
import zipfile
import os
import pandas as pd
from PIL import Image
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define paths
zip_file_path = '/Users/paramanandbhat/Downloads/train-scene classification.zip'
extraction_directory = '/Users/paramanandbhat/Downloads/train_scene'
extraction_directory_images = '/Users/paramanandbhat/Downloads/train_scene/train/'
test_csv_path = '/Users/paramanandbhat/Downloads/test_hAjxzwh.csv'
images_dir = '/Users/paramanandbhat/Downloads/train_scene/train/'
output_csv_path = '/Users/paramanandbhat/Downloads/train_scene/output_predictions.csv'

# Create the extraction directory if it doesn't exist
if not os.path.exists(extraction_directory):
    os.makedirs(extraction_directory)

# Unzip the dataset
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_directory)
print("Extraction completed.")

# Load the CSV file containing the labels
csv_file_path = os.path.join(extraction_directory, 'train.csv')
data = pd.read_csv(csv_file_path)
print(data.head())


extraction_directory_images = '/Users/paramanandbhat/Downloads/train_scene/train/'

#  image files are directly in the extraction directory, update if they are in a subdirectory
def load_image(image_name):
    image_path = os.path.join(extraction_directory_images, image_name)
    with Image.open(image_path) as img:
        return np.array(img)
    
# Load the images based on the image_name column in the CSV
data['image'] = data['image_name'].apply(load_image)

print("Data loading completed.")



from skimage.transform import resize
# Function to normalize and resize an image
def preprocess_image(image, target_size=(128, 128)):
    """
    Normalize pixel values to the range 0-1 and resize the image to a uniform size.
    
    Parameters:
    - image: numpy array, the image to be processed.
    - target_size: tuple, the target size (width, height) of the image.
    
    Returns:
    - Processed image as a numpy array.
    """
    # Normalize pixel values to 0-1
    normalized_image = image.astype('float32') / 255.0
    
    # Resize image
    resized_image = resize(normalized_image, target_size, anti_aliasing=True)
    
    return resized_image

# Apply the preprocessing function to each image in the DataFrame
data['processed_image'] = data['image'].apply(lambda x: preprocess_image(x))

print("Data pre-processing completed.")
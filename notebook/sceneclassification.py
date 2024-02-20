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

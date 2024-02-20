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

# Sample data simulation (replace with actual 'data' DataFrame in your environment)
categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
images_per_category = 2  # Number of images to display per category
data_demo = pd.DataFrame(data)

# Plotting
fig, axes = plt.subplots(len(categories), images_per_category, figsize=(10, 15))

for i, category in enumerate(categories):
    # Filter the dataset for the current category
    category_data = data_demo[data_demo['label'] == i].head(images_per_category)
    
    for j, (_, row) in enumerate(category_data.iterrows()):
        ax = axes[i, j]
        ax.imshow(row['processed_image'], cmap='gray')
        ax.axis('off')
        
        if j == 0:
            ax.set_title(category)

plt.tight_layout()
plt.show()

# Analyzing the distribution of classes
class_distribution = data['label'].value_counts().sort_index()

# Plotting the distribution
plt.figure(figsize=(10, 6))
class_distribution.plot(kind='bar')
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.title('Distribution of Classes')
plt.xticks(ticks=range(len(categories)), labels=categories, rotation=45)
plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.show()

# Checking for imbalances
print("Class distribution:\n", class_distribution)

imbalance_check = class_distribution.std() / class_distribution.mean()
print(f"\nImbalance metric (std/mean): {imbalance_check:.2f}")
if imbalance_check > 0.5:
    print("Significant class imbalance detected. Consider using strategies like oversampling or undersampling.")
else:
    print("Minor imbalances detected. The dataset is relatively balanced.")

# Define the model
model = Sequential([
    # Convolutional layer with 32 filters, a kernel size of 3, ReLU activation, and input shape defined
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # Update input_shape based on your data
    MaxPooling2D(2, 2),
    
    # Second convolutional layer with 64 filters
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Third convolutional layer with 128 filters
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Flatten the output of the convolutional layers
    Flatten(),
    
    # Dense (fully connected) layer with dropout for regularization
    Dense(128, activation='relu'),
    Dropout(0.5),
    
    # Output layer with softmax activation for multi-class classification
    Dense(6, activation='softmax')  # Update the number of neurons to match the number of classes in your dataset
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()


##https://datahack.analyticsvidhya.com/contest/assignment-scene-classification-challenge/download/train-file
##https://datahack.analyticsvidhya.com/contest/assignment-scene-classification-challenge/download/test-file


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
labels = to_categorical(data['label'])

# Splitting the dataset into training and temp (validation + test) sets
X_train, X_temp, y_train, y_temp = train_test_split(
    np.array(data_demo['processed_image'].tolist()),  # Ensuring the image data is in the correct format
    labels,  # Using the potentially one-hot encoded labels
    test_size=0.4,
    random_state=42,
    stratify=labels
)

# Splitting the temp set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# Number of epochs to train the model
epochs = 10

# Batch size for training
batch_size = 32

# Training the model
history = model.fit(
    X_train,  # Training data
    y_train,  # Labels for the training data
    epochs=epochs,  # Number of epochs
    batch_size=batch_size,  # Batch size
    validation_data=(X_val, y_val),  # Validation data
    verbose=1  # Show training log
)


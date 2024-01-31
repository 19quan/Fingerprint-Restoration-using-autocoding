import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#real_path = 'dataset/Real'
#altered_path = 'dataset/Altered/Altered-Easy'

altered_path = 'dataset/Altered/altered_fingerprint_specific'

altered_images = []

# Function to read images from a folder
def read_images_from_folder(folder_path, image_list):
    # Get a list of files in the folder
    file_list = os.listdir(folder_path)

    # Sort the file list to ensure consistent order
    file_list = sorted(file_list)
    
    
    # Loop through the files and read the first 100 images
    for filename in file_list:
        
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Read the image using cv2.imread
        image = cv2.imread(file_path)
        resized_image = cv2.resize(image, (100, 100))
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        
        if image is not None:
            # Append the image to the provided image_list
            image_list.append(grayscale_image)


# Read images from the "Altered-Easy" folder
print("loading images")
read_images_from_folder(altered_path, altered_images)
print("loading done")

# Normalize the images in the lists
altered_images = [image / 255.0 for image in altered_images]

altered_images_arr = np.array(altered_images)

altered_images_arr = altered_images_arr.astype('float32')

loaded_model = load_model('specific_trained.h5') # CHANGE MODEL USED TO PREDICT

reconstructed_fingerprint = loaded_model.predict(altered_images_arr)


# Display the input and reconstructed fingerprints
num_display = 9  # Number of fingerprints to display
fig, axes = plt.subplots(2, num_display, figsize=(24, 8))

for i in range(num_display):
    # Display the altered input fingerprint
    axes[0, i].imshow(altered_images_arr[i], cmap='gray')
    axes[0, i].set_title('Altered Image')
    axes[0, i].axis('off')
    
    # Rescale the reconstructed fingerprint from [0, 1] to [0, 255]
    reconstructed_fp = reconstructed_fingerprint[i] * 255.0
    
    # Convert the fingerprint to uint8 format
    reconstructed_fp = reconstructed_fp.astype(np.uint8)
    
    # Display the reconstructed fingerprint
    axes[1, i].imshow(reconstructed_fp, cmap='gray')
    axes[1, i].set_title('Restored Image')
    axes[1, i].axis('off')


# Save reconstructed fingerprints
save_path = 'results/reconstructed_fingerprints'
os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist

for i, fp in enumerate(reconstructed_fingerprint):
    # Rescale the reconstructed fingerprint from [0, 1] to [0, 255]
    reconstructed_fp = fp * 255.0
    
    # Convert the fingerprint to uint8 format
    reconstructed_fp = reconstructed_fp.astype(np.uint8)
    
    # Save the reconstructed fingerprint
    img_path = os.path.join(save_path, f"reconstructed_fingerprint_{i}.png")
    cv2.imwrite(img_path, reconstructed_fp)

plt.tight_layout()
plt.show()
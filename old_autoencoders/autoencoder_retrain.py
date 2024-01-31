import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from keras.optimizers import RMSprop
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

real_path = 'dataset/real_fingerprint1000'
altered_path = 'dataset/altered_fingerprint1000'

real_images = []
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
read_images_from_folder(real_path, real_images)
read_images_from_folder(altered_path, altered_images)
print("loading done")


# Normalize the images in the lists
real_images = [image / 255.0 for image in real_images]
altered_images = [image / 255.0 for image in altered_images]

real_images_arr = np.array(real_images)
altered_images_arr = np.array(altered_images)

real_images_arr = real_images_arr.astype('float32')
real_images_arr = real_images_arr.reshape(-1, 100, 100, 1)

altered_images_arr = altered_images_arr.astype('float32')
altered_images_arr = altered_images_arr.reshape(-1, 100, 100, 1)

X_train, X_test, Y_train, Y_test = train_test_split(altered_images_arr, real_images_arr, test_size=0.2, random_state=42)

loaded_model = load_model('test1.h5')


loaded_model.compile(optimizer=RMSprop(), loss='mse')

loaded_model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_test, Y_test))

loaded_model.save('test1.h5')


# Display original images
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(100, 100), cmap='gray')
    plt.title("Original")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Encoded images (you can use encoded_images or the autoencoder itself)
    ax = plt.subplot(3, n, i + 1 + n)
    encoded_img = loaded_model(X_test[i].reshape(1, 100, 100, 1))
    plt.imshow(encoded_img[0].numpy().reshape(100, 100), cmap='gray')
    plt.title("Encoded")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Decoded images
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    decoded_img = loaded_model.predict(X_test[i].reshape(1, 100, 100, 1))
    plt.imshow(decoded_img[0].reshape(100, 100), cmap='gray')
    plt.title("Reconstructed")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
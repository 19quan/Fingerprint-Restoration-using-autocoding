import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#real_path = 'dataset/Real'
#altered_path = 'dataset/Altered/Altered-Easy'

real_path = 'dataset/real_fingerprint'
altered_path = 'dataset/altered_fingerprint'

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

# Read images from the "Real" folder
read_images_from_folder(real_path, real_images)

# Read images from the "Altered-Easy" folder
read_images_from_folder(altered_path, altered_images)

# Normalize the images in the lists
real_images = [image / 255.0 for image in real_images]
altered_images = [image / 255.0 for image in altered_images]

# Create labels for real images (1 for real)
real_labels = [1] * len(real_images)

# Create labels for altered images (0 for altered)
altered_labels = [0] * len(altered_images)

# Combine real and altered data and labels
all_images = np.array(real_images + altered_images)
all_labels = np.array(real_labels + altered_labels)

real_images_arr = np.array(real_images)


X_train, X_test, ground_train, ground_test = train_test_split(all_images, all_images, test_size=0.2, random_state=42)

# Input layer
input_img = tf.keras.layers.Input(shape=(100, 100, 1))

def autoencoder(input_img):
    # Encoder network
    # Convert images into a compressed, encoded representation
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder network
    # Convert compressed representation into an approximation of original data
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded

# Compile model
autoencoder = tf.keras.models.Model(input_img, autoencoder(input_img))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse', 'mae'])

autoencoder.fit(X_train, ground_train, epochs=10, batch_size=32, validation_data=(X_test,ground_test))


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
    encoded_img = autoencoder(X_test[i].reshape(1, 100, 100, 1))
    plt.imshow(encoded_img[0].numpy().reshape(100, 100), cmap='gray')
    plt.title("Encoded")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Decoded images
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    decoded_img = autoencoder.predict(X_test[i].reshape(1, 100, 100, 1))
    plt.imshow(decoded_img[0].reshape(100, 100), cmap='gray')
    plt.title("Reconstructed")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
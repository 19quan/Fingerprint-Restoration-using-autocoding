import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import RMSprop
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#real_path = 'dataset/Real'
#altered_path = 'dataset/Altered/Altered-Easy'

real_path = 'dataset/real_fingerprint6000'
altered_path = 'dataset/altered_fingerprint6000'

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

real_images_arr = np.array(real_images)
altered_images_arr = np.array(altered_images)

real_images_arr = real_images_arr.astype('float32')
real_images_arr = real_images_arr.reshape(-1, 100, 100, 1)

altered_images_arr = altered_images_arr.astype('float32')
altered_images_arr = altered_images_arr.reshape(-1, 100, 100, 1)

X_train, X_test, Y_train, Y_test = train_test_split(altered_images_arr, real_images_arr, test_size=0.2, random_state=42)

# Input layer
input_img = tf.keras.layers.Input(shape=(100, 100, 1))

def autoencoder(input_img):
    #encoder
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)

    #decoder
    conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    up1 = tf.keras.layers.UpSampling2D((2,2))(conv4)
    conv5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    up2 = tf.keras.layers.UpSampling2D((2,2))(conv5)
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)
    return decoded

# Compile model
autoencoder = tf.keras.models.Model(input_img, autoencoder(input_img))

autoencoder.compile(optimizer=RMSprop(), loss='mse')

autoencoder.summary()

history = autoencoder.fit(X_train, Y_train, epochs=100, batch_size=128, validation_data=(X_test, Y_test))

autoencoder.save('my_model.h5')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

decoded_images = []

# Display original images
n = 5  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(100, 100), cmap='gray')
    plt.title("Original")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Decoded images
    ax = plt.subplot(3, n, i + 1 + n)
    decoded_img = autoencoder.predict(X_test[i].reshape(1, 100, 100, 1))
    plt.imshow(decoded_img[0].reshape(100, 100), cmap='gray')
    plt.title("Reconstructed")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    decoded_images.append(decoded_img[0])

save_path = 'results/autoencoder'
os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist

for i, img in enumerate(decoded_images):
    img_path = os.path.join(save_path, f"decoded_image_{i}.png")
    cv2.imwrite(img_path, img * 255)  # Rescale to 0-255 before saving as an image


plt.show()
import os
import cv2
from skimage.metrics import structural_similarity as ssim

# Function to read and resize images from a folder
def read_and_resize_images(folder_path, target_size=(100, 100)):
    image_list = []
    file_list = os.listdir(folder_path)
    file_list = sorted(file_list)
    
    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # Resize the image to the target size
            resized_image = cv2.resize(image, target_size)
            image_list.append(resized_image)
    
    return image_list

# Function to calculate SSIM between two images
def calculate_ssim(original_img, reconstructed_img):
    return ssim(original_img, reconstructed_img)

# Paths to original and reconstructed fingerprint images
original_path = 'dataset/Real/real_fingerprint_train'  # path to original fingerprint
reconstructed_path = 'results/reconstructed_fingerprints'  # Path where reconstructed images are saved

# Read and resize original and reconstructed images
original_images = read_and_resize_images(original_path)
reconstructed_images = read_and_resize_images(reconstructed_path)

# Ensure the number of images in both sets is the same
num_images = min(len(original_images), len(reconstructed_images))

# Calculate SSIM for each pair of images
ssim_values = []
for i in range(num_images):
    ssim_val = calculate_ssim(original_images[i], reconstructed_images[i])
    ssim_values.append(ssim_val)

# Average SSIM across all images
average_ssim = sum(ssim_values) / len(ssim_values)

# Print SSIM values and average SSIM
print("SSIM values for individual images:")
for i, val in enumerate(ssim_values):
    print(f"Image {i+1}: {val}")

print("\nAverage SSIM across all images:", average_ssim)

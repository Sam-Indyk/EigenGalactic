

import numpy as np
from sklearn.decomposition import IncrementalPCA
from PIL import Image
import requests
from io import BytesIO
import zipfile
import os
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from galaxy_datasets import gz2 
from tqdm import tqdm # For a nice progress bar in notebooks

# --- CONFIGURATION ---
IMAGE_SIZE = (64, 64)
NUM_TRAINING_IMAGES = 5000  
BATCH_SIZE = 100  
NUM_COMPONENTS = 100  
TEST_IMAGE_INDEX = 5

# === HOW TO USE YOUR OWN IMAGE ===
# 1. Place your image file in the same directory as this script.
# 2. Set the variable below to your image's filename.
# 3. If left as "", a random image from the dataset will be used.
CUSTOM_TEST_IMAGE_PATH = "" 

# --- UTILITY FUNCTIONS ---

def load_galaxy_zoo_data(size, num_images):
    """
    Loads images from the Galaxy Zoo 2 dataset.
    If a cache file of prepared data exists, it loads from there.
    Otherwise, it processes the raw images and creates the cache file.
    """
    data_root = 'galaxy_data'
    cache_file = os.path.join(data_root, f"cache_{num_images}_{size[0]}x{size[1]}.npy")

    # --- Caching Logic ---
    if os.path.exists(cache_file):
        print(f"Loading prepared image data from cache file: '{cache_file}'...")
        return np.load(cache_file)

    print("No cache file found. Preparing images for the first time...")
    print("Loading Galaxy Zoo 2 dataset via galaxy-datasets library...")
    try:
        catalog, _ = gz2(root=data_root, train=True, download=True)
    except Exception as e:
        print(f"Failed to download or load dataset. Error: {e}")
        return None

    # Process images in batches to save memory
    batch_size = 1000
    num_to_process = min(num_images, len(catalog))
    print(f"Preparing {num_to_process} images... (this may take a few minutes)")
    
    # Create the output array with the correct shape
    prepared_data = np.zeros((num_to_process, size[0] * size[1]), dtype=np.float32)
    
    for batch_start in tqdm(range(0, num_to_process, batch_size), desc="Processing image batches"):
        batch_end = min(batch_start + batch_size, num_to_process)
        batch_images = []
        
        for file_path in catalog['file_loc'][batch_start:batch_end]:
            try:
                # Newer galaxy-datasets versions already include data_root in file_loc.
                full_image_path = file_path if os.path.exists(file_path) else os.path.join(data_root, file_path)
                img = Image.open(full_image_path)
                img = img.convert('L')
                img = img.resize(size, Image.Resampling.LANCZOS)
                batch_images.append(np.array(img).flatten().astype(np.float32))
            except Exception as e:
                print(f"Could not process {full_image_path}: {e}")
                # If an image fails, use zeros as a placeholder
                batch_images.append(np.zeros(size[0] * size[1], dtype=np.float32))
        
        # Store the batch in the prepared data array
        prepared_data[batch_start:batch_end] = batch_images
        # Clear the batch_images list to free memory
        del batch_images
    
    print(f"Saving prepared data to cache file: '{cache_file}' for faster startup next time.")
    np.save(cache_file, prepared_data)
    
    return prepared_data


def create_gaussian_kernel(sigma, size=15):
    """Creates a 2D Gaussian kernel."""
    half = size // 2
    x, y = np.mgrid[-half:half+1, -half:half+1]
    kernel = np.exp(-(x**2 + y**2) / (2. * sigma**2))
    return kernel / kernel.sum()

# --- MAIN EXECUTION ---

def main():
    # --- Step 1: Get the Data and Prepare Training/Test Sets ---
    all_images_flat = load_galaxy_zoo_data(IMAGE_SIZE, NUM_TRAINING_IMAGES)
    if all_images_flat is None or all_images_flat.shape[0] == 0:
        print("No images were loaded. Exiting.")
        return

    # Prepare the test image and training data
    if CUSTOM_TEST_IMAGE_PATH and os.path.exists(CUSTOM_TEST_IMAGE_PATH):
        print(f"\nLoading custom test image from: {CUSTOM_TEST_IMAGE_PATH}")
        img = Image.open(CUSTOM_TEST_IMAGE_PATH).convert('L')
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        original_sharp_image = np.array(img)
        # The entire dataset will be used for training
        training_data = all_images_flat
    else:
        if CUSTOM_TEST_IMAGE_PATH:
             print(f"\nWARNING: Custom image '{CUSTOM_TEST_IMAGE_PATH}' not found. Using an image from the dataset instead.")
        # Separate a test image from the dataset
        test_image_flat = all_images_flat[TEST_IMAGE_INDEX]
        training_data = np.delete(all_images_flat, TEST_IMAGE_INDEX, axis=0)
        original_sharp_image = test_image_flat.reshape(IMAGE_SIZE)
        
    # --- Step 2: Learn the Eigengalaxies using IncrementalPCA ---
    print(f"\n--- Step 2: Learning {NUM_COMPONENTS} Eigengalaxies using IncrementalPCA ---")
    
    # Use IncrementalPCA for memory-efficient training on large datasets.
    pca = IncrementalPCA(n_components=NUM_COMPONENTS, batch_size=BATCH_SIZE)
    
    # Fit the model in batches
    for i in tqdm(range(0, len(training_data), BATCH_SIZE), desc="Training PCA model"):
        batch = training_data[i:i + BATCH_SIZE]
        pca.partial_fit(batch)
        
    eigengalaxies = pca.components_
    
    # --- Interactive Menu ---
    print("\n--- Main Menu ---")
    print("1: Run the full Eigengalaxy reconstruction pipeline.")
    print("2: Analyze and display the Eigengalaxy basis only.")
    choice = input("Please enter your choice (1 or 2): ")

    # --- Step 2a: Diagnostic Visualizations (Always Show) ---
    print("\n--- Displaying PCA Diagnostics ---")
    
    # Display the Mean Galaxy
    plt.figure(figsize=(6, 6))
    plt.imshow(pca.mean_.reshape(IMAGE_SIZE), cmap='gray')
    plt.title("Mean Galaxy", fontsize=16)
    plt.axis('off')
    plt.savefig("mean_galaxy.png")
    plt.show()

    # Display the first 10 Eigengalaxies
    fig_eigen, axes_eigen = plt.subplots(2, 5, figsize=(15, 6))
    fig_eigen.suptitle("Top 10 Eigengalaxies", fontsize=20)
    for i, ax in enumerate(axes_eigen.flat):
        ax.imshow(eigengalaxies[i].reshape(IMAGE_SIZE), cmap='gray')
        ax.set_title(f"Component {i+1}")
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("eigengalaxies.png")
    plt.show()

    # Plot the cumulative explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("How Many Eigengalaxies are Needed?", fontsize=16)
    plt.grid(True)
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% Variance Threshold')
    plt.legend()
    plt.ylim(0, 1.05)
    plt.savefig("explained_variance.png")
    plt.show()
    
    print("Explained variance with first component: {:.2f}%".format(pca.explained_variance_ratio_[0] * 100))
    print("Total variance explained by {} components: {:.2f}%".format(NUM_COMPONENTS, np.sum(pca.explained_variance_ratio_) * 100))

    # --- Conditional Execution based on menu choice ---
    if choice != '1':
        print("\nAnalysis complete. Exiting.")
        return

    # --- Step 3: Create the Blurry Test Image ---
    print("\n--- Step 3: Blurring the Test Image ---")
    blur_kernel = create_gaussian_kernel(sigma=2.5)
    blurry_image = convolve2d(original_sharp_image, blur_kernel, 'same', boundary='symm')

    # --- Step 4: Reconstruct the Image ---
    print("\n--- Step 4: Reconstructing from Eigengalaxies ---")
    
    blurry_flat = blurry_image.flatten()
    
    # Use pca.transform to project the data onto the principal components
    # This automatically handles subtracting the mean
    coefficients = pca.transform(blurry_flat.reshape(1, -1))
    
    # Use pca.inverse_transform to reconstruct from the coefficients
    # This automatically handles adding the mean back
    reconstructed_flat = pca.inverse_transform(coefficients)
    
    reconstructed_image = reconstructed_flat.reshape(IMAGE_SIZE)

    # --- Step 5: Display Results ---
    print("\n--- Step 5: Displaying Comparison ---")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Eigengalaxy Reconstruction", fontsize=20)

    axes[0].imshow(original_sharp_image, cmap='gray')
    axes[0].set_title("1. Original Sharp Image", fontsize=16)
    axes[0].axis('off')

    axes[1].imshow(blurry_image, cmap='gray')
    axes[1].set_title("2. Blurry Image", fontsize=16)
    axes[1].axis('off')
    
    axes[2].imshow(reconstructed_image, cmap='gray')
    axes[2].set_title("3. Reconstructed from Basis", fontsize=16)
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("eigengalaxy_reconstruction.png")
    print("\nResult plot saved as 'eigengalaxy_reconstruction.png'")
    plt.show()

if __name__ == "__main__":
    main()


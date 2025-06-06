# Import the numpy library for working with arrays
import numpy as np

# Import torch for deep learning operations (needed for torchvision)
import torch

# Import the STL10 image dataset from torchvision
from torchvision.datasets import STL10

# Import transforms to help convert images to tensors
from torchvision import transforms

# Import MobileNetV2 model from TensorFlow Keras for feature extraction
from tensorflow.keras.applications import MobileNetV2

# Import the preprocessing function for MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Import MinHash and MinHashLSH for fast similarity searching
from datasketch import MinHash, MinHashLSH

# Import matplotlib for displaying images
import matplotlib.pyplot as plt

# Import OpenCV for resizing images
import cv2

# ==================== STEP 1: Load 10,000 Images from STL10 ====================

# Set up a transform to convert PIL images to tensors
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the STL10 training dataset with the transform
train_set = STL10(root='./data', split='train', download=True, transform=transform)

# Download and load the STL10 test dataset with the transform
test_set = STL10(root='./data', split='test', download=True, transform=transform)

# Create empty lists to store images and their labels
images = []
labels = []

# Go through each image and label in the training set
for img, label in train_set:
    # Convert image from CHW (channel, height, width) to HWC (height, width, channel)
    img = np.transpose(img.numpy(), (1, 2, 0))
    # Scale pixel values from 0-1 to 0-255 and change type to uint8
    img = (img * 255).astype(np.uint8)
    # Add the processed image to the images list
    images.append(img)
    # Add the label to the labels list
    labels.append(label)

# Go through each image and label in the test set
for img, label in test_set:
    # Convert image from CHW to HWC
    img = np.transpose(img.numpy(), (1, 2, 0))
    # Scale pixel values to 0-255 and convert to uint8
    img = (img * 255).astype(np.uint8)
    # Add the processed image to the list
    images.append(img)
    # Add the label to the labels list
    labels.append(label)

# Only keep the first 10,000 images and labels
images = np.stack(images[:10000])  # Combine the list into a numpy array
labels = np.array(labels[:10000])  # Convert labels list to a numpy array
label_names = train_set.classes     # Get the list of class names

# Print the shape of the images array (should be 10000, 96, 96, 3)
print("Loaded images shape:", images.shape)
# Print the shape of the labels array (should be 10000,)
print("Loaded labels shape:", labels.shape)

# ==================== STEP 2: Resize and Preprocess for MobileNetV2 ====================

# Resize each image to 96x96 pixels (expected input size for MobileNetV2)
images_resized = np.array([cv2.resize(img, (96, 96), interpolation=cv2.INTER_NEAREST) for img in images])

# Preprocess the images for MobileNetV2 (scales pixel values to the right range)
images_pre = preprocess_input(images_resized)

# ==================== STEP 3: Extract CNN Features ====================

# Load the MobileNetV2 model without the final classification layer
# 'include_top=False' removes the last layer, so we get features, not predictions
# 'pooling="avg"' means we take the average of features (global average pooling)
# 'input_shape' should match our image size (96, 96, 3)
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(96, 96, 3))

# Get feature vectors for all images using the model
# 'batch_size=32' means process 32 images at a time
# 'verbose=1' shows a progress bar
features = model.predict(images_pre, batch_size=32, verbose=1)

# Print the shape of the extracted features (should be 10000, 1280)
print("Feature shape:", features.shape)

# ==================== STEP 4: Build LSH Index on CNN Features ====================

# Create a Locality Sensitive Hashing (LSH) index for fast similarity search
# 'threshold=0.3' means only consider items with Jaccard similarity above 0.3
# 'num_perm=64' sets the number of hash functions (affects accuracy & speed)
lsh = MinHashLSH(threshold=0.50, num_perm=64)

# List to store MinHash signatures of each image
minhashes = []
# List to store the set of the top 50 feature indices for each image
topk_idxs_list = []

# Go through each image's features
for i, feature in enumerate(features):
    # Find the indices of the top 50 largest values in the feature vector
    idxs = np.argsort(feature)[-50:]
    # Store these indices as a set for that image
    topk_idxs_list.append(set(idxs))

    # Create a new MinHash object (sketch) for this image
    m = MinHash(num_perm=64)
    # Update the MinHash with each top feature index (converted to string, then bytes)
    for idx in idxs:
        m.update(str(idx).encode('utf8'))
    # Add this MinHash to the LSH index with a label (e.g., "img_5")
    lsh.insert(f"img_{i}", m)
    # Save the MinHash for later querying
    minhashes.append(m)

# ==================== STEP 5: Query LSH for Similar Images ====================

# Choose the index of the query image (change this number to try different images)
query_idx = 17

# Ask the LSH to find images similar to the query image by comparing their MinHash sketches
result = lsh.query(minhashes[query_idx])

# Print the IDs of images found to be similar to the query image
print(f"Images similar to image {query_idx}: {result}")

# ==================== STEP 6: Set Operations Between Query and Similar Image ====================

# Function to print intersections, unions, and differences between feature sets
def print_set_operations(query_idx, result, topk_idxs_list):
    # Remove the query image itself from the result list (don't compare to itself)
    filtered_result = [img_id for img_id in result if int(img_id.split('_')[1]) != query_idx]
    
    # If there are no other similar images, print a message and exit
    if not filtered_result:
        print("No similar images to compare for set operations.")
        return None
    
    # Get the index of the most similar image (the first one in the list)
    similar_idx = int(filtered_result[0].split('_')[1])

    # Get the sets of top features for the query and the similar image
    set_query = topk_idxs_list[query_idx]
    set_sim = topk_idxs_list[similar_idx]

    # Find features shared by both images
    intersection = set_query & set_sim
    # Find all unique features from both images
    union = set_query | set_sim
    # Find features in the query image but not in the similar image
    difference = set_query - set_sim

    # Print set operations results in a readable way
    print(f"\nSet operations between image {query_idx} and its most similar image {similar_idx}:")
    print("Intersection (shared top features):", intersection)
    print("Union (all top features):", union)
    print("Difference (query - similar):", difference)
    # Print Jaccard similarity (size of intersection / size of union)
    print("Jaccard similarity:", len(intersection) / len(union))

    # Return the similar image's index for later use
    return similar_idx

# Call the function to print and compare features between the query and its most similar image
similar_idx = print_set_operations(query_idx, result, topk_idxs_list)

# ==================== STEP 7: Visualization ====================

# Function to show the query image and its most similar images side-by-side
def show_similar_images(query_idx, result, images, labels, label_names, num_to_show=5, scale=8, dpi=120, upscale_for_display=True):
    # Helper function to enlarge images for better viewing
    def upscale(img, size=(128, 128)):
        return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

    # Create a figure with space for query plus several similar images
    plt.figure(figsize=(scale * (num_to_show + 1) // 2, scale), dpi=dpi)

    # Show the query image in the first subplot
    plt.subplot(1, num_to_show + 1, 1)
    img_to_show = images[query_idx]
    if upscale_for_display:
        img_to_show = upscale(img_to_show)
    plt.imshow(img_to_show, interpolation='nearest')
    plt.title(f'Query\nLabel: {label_names[labels[query_idx]]}')
    plt.axis('off')

    # Show the top similar images in the next subplots
    for i, img_id in enumerate(result[:num_to_show]):
        idx = int(img_id.split('_')[1])  # Get the image index from the label string
        plt.subplot(1, num_to_show + 1, i + 2)
        img_to_show = images[idx]
        if upscale_for_display:
            img_to_show = upscale(img_to_show)
        plt.imshow(img_to_show, interpolation='nearest')
        plt.title(f"{img_id}\nLabel: {label_names[labels[idx]]}")
        plt.axis('off')

    # Make layout look good
    plt.tight_layout()
    # Show the images on the screen
    plt.show()

# Call the function to display the query image and its most similar images
show_similar_images(query_idx, result, images, labels, label_names)
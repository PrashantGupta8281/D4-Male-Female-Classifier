
# -*- coding: utf-8 -*-
import os
import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Dataset path
dataset_path = r"/content/sample_data"

images = []
labels = []
classes = ["Male", "Female"]
IMG_SIZE = 64
# Valid image extensions to prevent reading hidden/system files
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

print("Loading dataset...")

for label, folder in enumerate(classes):
    folder_path = os.path.join(dataset_path, folder)
    
    # Check if the folder actually exists to avoid FileNotFound errors
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found at {folder_path}")
        continue

    for file in os.listdir(folder_path):
        # Filter out non-image files (.DS_Store, text files, etc.)
        if not file.lower().endswith(VALID_EXTENSIONS):
            continue

        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Resize and flatten the image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.flatten()

        images.append(img)
        labels.append(label)

# Convert lists to NumPy arrays
X = np.array(images)
y = np.array(labels)

# Check if any images were successfully loaded
if len(X) == 0:
    raise ValueError("No images were loaded! Please check your dataset path and structure.")

print(f"Training Images loaded: {len(X)}")

# --- CRITICAL FIX: Feature Scaling ---
# Scale pixel values from [0-255] to [0-1] for optimal model convergence
X = X / 255.0

print("Training the Logistic Regression model... (This might take a moment)")
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "Male_&_Female_model.pkl")
print("Model Saved Successfully as 'Male_&_Female_model.pkl'!")

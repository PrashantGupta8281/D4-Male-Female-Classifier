
import os
import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Define Dataset Path
dataset_path = r"/content/sample_data"
classes = ["Male", "Female"]
IMG_SIZE = 64
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

images = []
labels = []

print("Step 1: Checking and preparing directories...")
# Create directories automatically if they do not exist to prevent FileNotFoundError
for cl in classes:
    target_folder = os.path.join(dataset_path, cl)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"-> Created missing folder: {target_folder}")
        print(f"   Please upload your {cl} images inside this folder in Colab.")

print("\nStep 2: Loading and preprocessing images...")
# 2. Load and preprocess images safely
for label, folder in enumerate(classes):
    folder_path = os.path.join(dataset_path, folder)
    
    file_list = os.listdir(folder_path)
    if not file_list:
        print(f"Warning: '{folder}' folder is currently empty.")
        continue

    for file in file_list:
        # Strict extension check to avoid system files like .DS_Store or .ipynb_checkpoints
        if not file.lower().endswith(VALID_EXTENSIONS):
            continue

        img_path = os.path.join(folder_path, file)
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Resize and flatten image matrix into a 1D vector
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.flatten()

            images.append(img)
            labels.append(label)
        except Exception as e:
            # Skip any corrupt images dynamically without crashing the script
            print(f"Skipping damaged file {file}: {e}")
            continue

# 3. Validation Check before training
if len(images) == 0:
    print("\n[STOP] Error: No valid images found to train the model!")
    print(f"Please ensure you have uploaded image files into: {dataset_path}/Male and {dataset_path}/Female")
else:
    X = np.array(images)
    y = np.array(labels)

    print(f"\nSuccessfully loaded {len(X)} images.")
    print("Step 3: Normalizing features...")
    # Normalize pixel data (0-255 -> 0.0-1.0) for optimal Logistic Regression convergence
    X = X / 255.0

    print("Step 4: Training Logistic Regression model...")
    # max_iter is set to 1000 to ensure the solver converges properly
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # 4. Save the trained model safely
    model_filename = "Male_&_Female_model.pkl"
    joblib.dump(model, model_filename)
    print(f"\n[SUCCESS] Model saved successfully as '{model_filename}'!")

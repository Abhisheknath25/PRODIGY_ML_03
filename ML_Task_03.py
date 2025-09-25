import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# -----------------------------
# CONFIGURATION
# -----------------------------
DATASET_DIR = r"C:\Users\natha\OneDrive\Desktop\prodigy 03\train"  # your dataset path
IMG_SIZE = 64                  # resize images
CATEGORIES = ["Cat", "Dog"]    # labels

# -----------------------------
# LOAD DATA FUNCTION
# -----------------------------
def load_data(dataset_dir, img_size=64, limit=None):
    X, y = [], []
    categories = os.listdir(dataset_dir)  # expects cats/, dogs/
    
    for label, category in enumerate(categories):
        category_path = os.path.join(dataset_dir, category)
        if not os.path.isdir(category_path):
            continue

        count = 0
        for file in tqdm(os.listdir(category_path), desc=f"Loading {category}"):
            try:
                img_path = os.path.join(category_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Failed to read {img_path}, skipping.")
                    continue
                img = cv2.resize(img, (img_size, img_size))
                X.append(img.flatten())
                y.append(label)

                count += 1
                if limit and count >= limit:
                    break
            except Exception as e:
                print("Error:", e)
                continue

    return np.array(X), np.array(y)

# -----------------------------
# PREDICT SINGLE IMAGE
# -----------------------------
def predict_image(model, img_path, img_size=64):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image path not found: {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    img = cv2.resize(img, (img_size, img_size))
    img_flat = img.flatten().reshape(1, -1)
    prediction = model.predict(img_flat)[0]
    return CATEGORIES[prediction]

# -----------------------------
# MAIN CODE
# -----------------------------
print("Loading dataset...")
X, y = load_data(DATASET_DIR, IMG_SIZE, limit=1000)  # use limit=None to load all images

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training SVM model...")
model = SVC(kernel="linear", C=1.0)
model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=CATEGORIES))

# -----------------------------
# TEST WITH A NEW IMAGE
# -----------------------------
test_img_path = r"C:\Users\natha\OneDrive\Desktop\prodigy 03\test\cats\cat.18.jpg"  # change this to your image
try:
    result = predict_image(model, test_img_path, IMG_SIZE)
    print(f"\nPrediction for {test_img_path}: {result}")
except Exception as e:
    print("Error:", e)

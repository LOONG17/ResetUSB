import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Function to extract features from an image
def extract_features(image_path):
    # Load image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Flatten the image into a 1D array
    flattened_image = gray.flatten()
    return flattened_image

# Prepare training data
# Assuming you have a directory with labeled images (dangerous and non-dangerous)
# Adjust the paths accordingly
dangerous_images = ['dangerous1.jpg', 'dangerous2.jpg', ...]
safe_images = ['safe1.jpg', 'safe2.jpg', ...]

X = []
y = []

# Extract features for dangerous images
for image_path in dangerous_images:
    features = extract_features(image_path)
    X.append(features)
    y.append(1)  # Label 1 for dangerous

# Extract features for safe images
for image_path in safe_images:
    features = extract_features(image_path)
    X.append(features)
    y.append(0)  # Label 0 for safe

# Train the classifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Save the trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)
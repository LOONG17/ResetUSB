import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def extract_features(image_path):
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flattened_image = gray.flatten()
    return flattened_image

# Prepare training data
dangerous_images = ['dangerous1.jpg', 'dangerous2.jpg', ...]
safe_images = ['safe1.jpg', 'safe2.jpg', ...]

X = []
y = []

for image_path in dangerous_images:
    features = extract_features(image_path)
    X.append(features)
    y.append(1)  

for image_path in safe_images:
    features = extract_features(image_path)
    X.append(features)
    y.append(0) 

clf = RandomForestClassifier()
clf.fit(X, y)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

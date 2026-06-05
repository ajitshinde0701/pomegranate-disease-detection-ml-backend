import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from utils.feature_extractor import extract_features

DATASET_PATH = "dataset"

features = []
labels = []

classes = os.listdir(DATASET_PATH)

print("Loading Dataset...")

for label in classes:

    folder = os.path.join(DATASET_PATH, label)

    for image_name in os.listdir(folder):

        image_path = os.path.join(folder, image_name)

        feature = extract_features(image_path)

        if feature is not None:

            features.append(feature)
            labels.append(label)

print("Dataset Loaded")

X = np.array(features)
y = np.array(labels)

print("Splitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Trained")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print(classification_report(y_test, predictions))

# Save Model
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/pomegranate_rf_model.pkl")

print("Model Saved")
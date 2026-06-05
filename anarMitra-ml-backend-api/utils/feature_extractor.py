import cv2
import numpy as np
from skimage.feature import hog

def extract_features(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    # Resize image
    image = cv2.resize(image, (128, 128))

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to Gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0)
    gray = clahe.apply(gray)

    # HOG Features
    hog_features = hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    # Color Histogram
    hist = cv2.calcHist(
        [image_rgb],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )

    hist = cv2.normalize(hist, hist).flatten()

    # Combine Features
    features = np.hstack([hog_features, hist])

    return features
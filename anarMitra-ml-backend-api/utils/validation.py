import cv2

def validate_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return False

    h, w = image.shape[:2]

    if h < 50 or w < 50:
        return False

    return True
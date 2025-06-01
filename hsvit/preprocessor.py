import cv2
import numpy as np

def skull_strip(image):
    """
    Perform skull stripping using thresholding and largest contour selection.
    """
    norm_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(norm_img, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return norm_img

    mask = np.zeros_like(norm_img)
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    stripped = cv2.bitwise_and(norm_img, norm_img, mask=mask)
    return stripped

def apply_clahe(image):
    """
    Enhance contrast using CLAHE (adaptive histogram equalization).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    enhanced = clahe.apply(norm_img)
    return enhanced

def gaussian_smooth(image, kernel_size=5):
    """
    Apply Gaussian blur to reduce noise and preserve edges.
    """
    norm_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    smoothed = cv2.GaussianBlur(norm_img, (kernel_size, kernel_size), 0)
    return smoothed


def preprocess_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline.
    """
    image = skull_strip(image)
    image = apply_clahe(image)
    image = gaussian_smooth(image)
    return image
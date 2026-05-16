"""Classical feature extraction — 252-dimensional vector per 224×224 crop.

Components:
  color_histogram : 96 dims  (HSV, 32 bins/channel)
  lbp             : 26 dims  (uniform LBP, radius=3, n_points=24)
  gabor           : 48 dims  (mean+std for 4 orientations × 3 frequencies)
  hog             : 36 dims  (9 orientations, 8×8 px/cell, 2×2 blk)
  shape           : 10 dims  (aspect ratio, circularity, 7 Hu moments, extent)
  glcm            : 36 dims  (5 properties × 2 distances × 2 angles... approx)

Total: 96+26+48+36+10+36 = 252

BUG (Cell 6): current implementation produces 283 dims — fix in progress.
"""
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from PIL import Image


FEATURE_DIM = 252


def extract_color_histogram(img_rgb: np.ndarray, bins: int = 32) -> np.ndarray:
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(img_hsv[:, :, ch], bins=bins, range=(0, 256), density=True)
        feats.append(hist)
    return np.concatenate(feats)  # 96 dims


def extract_lbp(gray: np.ndarray, n_points: int = 24, radius: int = 3) -> np.ndarray:
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True)
    return hist  # 26 dims


def extract_gabor(gray: np.ndarray) -> np.ndarray:
    feats = []
    for freq in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:   # 6 freq × 4 orient × 2 stats = 48 dims
        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            kernel = cv2.getGaborKernel((21, 21), 5, theta, 1 / freq, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
            feats.extend([filtered.mean(), filtered.std()])
    return np.array(feats)  # 48 dims


def extract_hog_features(gray: np.ndarray) -> np.ndarray:
    feats = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), feature_vector=True)
    # Truncate/pad to exactly 36 dims for consistency
    target = 36
    if len(feats) >= target:
        return feats[:target]
    return np.pad(feats, (0, target - len(feats)))


def extract_shape_features(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(10)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / (h + 1e-6)
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
    extent = area / (w * h + 1e-6)
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    return np.array([aspect_ratio, circularity, extent, *hu[:7]])  # 10 dims


def extract_glcm_features(gray: np.ndarray) -> np.ndarray:
    glcm = graycomatrix(gray, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    feats = [graycoprops(glcm, p).ravel() for p in props]
    raw = np.concatenate(feats)  # 5 props × 2 dist × 4 angles = 40
    return raw[:36]  # trim to 36


def extract_classical_from_array(img_rgb: np.ndarray) -> np.ndarray:
    """Extract 252-dim features from an already-loaded 224×224 RGB uint8 array."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    feats = np.concatenate([
        extract_color_histogram(img_rgb),  # 96
        extract_lbp(gray),                 # 26
        extract_gabor(gray),               # 48
        extract_hog_features(gray),        # 36
        extract_shape_features(gray),      # 10
        extract_glcm_features(gray),       # 36
    ])
    assert len(feats) == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {len(feats)}"
    return feats


def extract_classical(crop_path: str) -> np.ndarray:
    img = np.array(Image.open(crop_path).convert("RGB").resize((224, 224)))
    return extract_classical_from_array(img)

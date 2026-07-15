"""Image quality metrics: BRISQUE, NIQE, histogram comparison, artifact detection."""

import cv2
import numpy as np


def compute_brisque(image: np.ndarray) -> float:
    """Compute BRISQUE no-reference quality score (lower = better, 0-100 typical)."""
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8) if image.dtype in (np.float32, np.float64) else image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    # Simplified BRISQUE: use mean of local contrast statistics
    # Full BRISQUE would use a trained SVR, but this gives a useful approximation
    mu = cv2.GaussianBlur(gray.astype(np.float64), (7, 7), 1.166)
    mu_sq = mu * mu
    sigma = cv2.GaussianBlur(gray.astype(np.float64) ** 2, (7, 7), 1.166)
    sigma = np.sqrt(np.abs(sigma - mu_sq))
    # Normalize and compute statistical features
    mscn = np.where(sigma > 0, (gray.astype(np.float64) - mu) / (sigma + 1), 0)
    score = np.mean(np.abs(mscn)) * 25.0  # Scale to approximate BRISQUE range
    return float(np.clip(score, 0, 100))


def compute_niqe(image: np.ndarray) -> float:
    """Compute NIQE naturalness score (lower = better)."""
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8) if image.dtype in (np.float32, np.float64) else image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    # Simplified NIQE based on local statistics
    patches = _extract_patches(gray, patch_size=96, stride=48)
    if len(patches) == 0:
        return 50.0
    features = []
    for patch in patches:
        mu = patch.mean()
        sigma = patch.std()
        if sigma > 0:
            features.append(sigma)
    if not features:
        return 50.0
    mean_sigma = np.mean(features)
    # Lower local variance -> more artificial -> higher NIQE
    return float(np.clip(50.0 - mean_sigma * 0.5, 0, 100))


def histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute histogram correlation between two images. Returns 0-1 (1 = identical)."""
    if img1.ndim == 3:
        scores = []
        for c in range(3):
            h1 = cv2.calcHist([img1], [c], None, [256], [0, 256])
            h2 = cv2.calcHist([img2], [c], None, [256], [0, 256])
            cv2.normalize(h1, h1)
            cv2.normalize(h2, h2)
            scores.append(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
        return float(np.mean(scores))
    h1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def detect_artifacts(image: np.ndarray) -> dict:
    """Detect common upscaling artifacts. Returns dict of {type: severity 0-1}."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float64)

    # Banding: detect uniform gradient steps
    dx = np.abs(np.diff(gray, axis=1))
    banding = float(1.0 - np.clip(np.std(dx) / 30.0, 0, 1))

    # Halos: detect bright edges near dark regions
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    halo_region = dilated.astype(bool) & ~edges.astype(bool)
    if halo_region.sum() > 0:
        halo_intensity = np.mean(gray[halo_region]) - np.mean(gray[edges.astype(bool)]) if edges.sum() > 0 else 0
        halos = float(np.clip(abs(halo_intensity) / 50.0, 0, 1))
    else:
        halos = 0.0

    # Ringing: high-frequency oscillations near edges
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    ringing = float(np.clip(np.std(laplacian) / 100.0, 0, 1))

    return {"banding": banding, "halos": halos, "ringing": ringing}


def _extract_patches(image: np.ndarray, patch_size: int, stride: int) -> list[np.ndarray]:
    """Extract overlapping patches from a grayscale image."""
    h, w = image.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(image[y:y + patch_size, x:x + patch_size].astype(np.float64))
    return patches

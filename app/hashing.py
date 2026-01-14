"""
Image hashing and face matching functions for tile identity detection.
"""

import cv2
import numpy as np


def normalize_face(face):
    """
    Normalize a face image for consistent hashing.
    Converts to grayscale, resizes, equalizes histogram, and blurs.
    """
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))

    # Remove lighting differences
    gray = cv2.equalizeHist(gray)

    # Kill high-frequency autofocus noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray


def dhash64(img, hash_size=8):
    """
    Compute difference hash (dHash) for an image.
    Accepts either BGR (3-ch) or grayscale (1-ch) image.
    Returns a 64-bit dHash as Python int.
    """
    if img is None or img.size == 0:
        return 0

    # If already grayscale, use it directly
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to (hash_size+1, hash_size) so we can diff adjacent columns
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]

    h = 0
    for bit in diff.flatten():
        h = (h << 1) | int(bit)
    return h


def hamming64(a, b):
    """Compute Hamming distance between two 64-bit hashes."""
    return (a ^ b).bit_count()


def match_identity(new_hash, existing_hash_by_tile, max_dist=10):
    """
    Find a matching tile for the given hash.
    Returns (matched_tile_idx, dist) or (None, None).
    """
    best_tile = None
    best_d = 1e9
    for ti, h in existing_hash_by_tile.items():
        d = hamming64(new_hash, h)
        if d < best_d:
            best_d = d
            best_tile = ti
    if best_d <= max_dist:
        return best_tile, best_d
    return None, None


def deterministic_color(idx):
    """Generate a deterministic bright BGR color from an index."""
    rng = np.random.default_rng(idx)
    c = rng.integers(low=60, high=255, size=3, dtype=np.int32)
    return (int(c[0]), int(c[1]), int(c[2]))

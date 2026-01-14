"""
Geometric helper functions for coordinate transformations and quad operations.
"""

import cv2
import numpy as np


def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def angle_cos(p0, p1, p2):
    """Cosine of angle at p1 formed by p0-p1-p2 (closer to 0 => closer to 90 degrees)."""
    v1 = p0 - p1
    v2 = p2 - p1
    return abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))


def quad_diag(pts):
    """Return diagonal length of quad (top-left to bottom-right)."""
    pts = order_points(pts.astype(np.float32))
    return float(np.linalg.norm(pts[2] - pts[0]))


def quad_area(pts):
    """Calculate polygon area using shoelace formula."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def iou_like(a, b):
    """Cheap similarity measure: average corner distance normalized."""
    return np.mean(np.linalg.norm(a - b, axis=1))


def smooth_quad(prev, new, alpha=0.85):
    """Smooth quad position with exponential moving average. Alpha close to 1 => more stable."""
    return (alpha * prev + (1 - alpha) * new).astype(np.float32)


def four_point_warp(image, pts):
    """Perform perspective transform to get a top-down view of quad."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    # Prevent tiny/invalid warps
    if maxW < 50 or maxH < 50:
        return None

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def sanitize_boxes(boxes):
    """Convert box coordinates to clean integers."""
    clean = []
    for (x, y, w, h) in boxes:
        clean.append((int(round(x)), int(round(y)), int(round(w)), int(round(h))))
    return clean

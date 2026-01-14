"""
Screen and tile detection functions using computer vision.
"""

import cv2
import numpy as np

from config import DEBUG
from geometry import order_points, angle_cos


def find_screen(frame):
    """
    Detect iPad screen in the camera frame.
    Returns 4 corner points or None if not found.
    """
    h, w = frame.shape[:2]

    # BRIGHT SCREEN MASK: prefer the lit screen
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    v_blur = cv2.GaussianBlur(v, (5, 5), 0)

    # Otsu chooses threshold automatically; keep pixels >= threshold (bright)
    t, _ = cv2.threshold(v_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.inRange(v_blur, int(t), 255)

    # Fill holes caused by text/UI so the screen becomes a solid blob
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Clean small specks
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    if DEBUG:
        cv2.imshow("debug mask", mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:40]

    # Debug: show top contour
    dbg = frame.copy()
    cv2.drawContours(dbg, cnts[:1], -1, (0, 255, 255), 3)
    if DEBUG:
        cv2.imshow("debug top contour", dbg)

    best = None
    best_score = -1.0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 12000:
            continue

        # minAreaRect fill ratio
        rot_rect = cv2.minAreaRect(c)
        (box_w, box_h) = rot_rect[1]
        if box_w < 1 or box_h < 1:
            continue
        fill = float(area) / float(box_w * box_h + 1e-6)
        if fill < 0.70:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        pts = approx.reshape(4, 2).astype(np.float32)
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        # Reject candidates hugging the frame boundary
        margin = 10
        if (rect[:, 0].min() < margin or rect[:, 1].min() < margin or
            rect[:, 0].max() > (w - margin) or rect[:, 1].max() > (h - margin)):
            continue

        # Angle check: prefer near-90-degree corners
        cos1 = angle_cos(tl, tr, br)
        cos2 = angle_cos(tr, br, bl)
        cos3 = angle_cos(br, bl, tl)
        cos4 = angle_cos(bl, tl, tr)
        max_cos = max(cos1, cos2, cos3, cos4)

        if max_cos > 0.35:
            continue
        angle_score = max(0.0, 1.0 - (max_cos / 0.35))

        # Width/height from side lengths
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        width = max(widthA, widthB)
        height = max(heightA, heightB)
        if width < 80 or height < 80:
            continue

        # Aspect ratio
        aspect = width / height
        aspect = aspect if aspect >= 1 else 1 / aspect
        if aspect > 2.2:
            continue

        # iPad-ish aspect scoring
        aspect_target = 1.33
        aspect_sigma = 0.28
        aspect_score = float(np.exp(-((aspect - aspect_target) ** 2) / (2 * aspect_sigma ** 2)))

        # Center bias
        cx = float(np.mean(rect[:, 0]))
        cy = float(np.mean(rect[:, 1]))
        center_dist = np.linalg.norm([cx - w/2, cy - h/2]) / (np.linalg.norm([w/2, h/2]) + 1e-6)
        center_score = float(1.0 - center_dist)

        # Area fraction
        area_frac = float(area) / float(w * h + 1e-6)

        score = (
            3.0 * area_frac +
            3.0 * fill +
            3.0 * aspect_score +
            2.0 * angle_score +
            0.5 * center_score
        )

        if score > best_score:
            best_score = score
            best = pts

    return best


def detect_spiral_centers(warped_bgr):
    """Detect spiral centers in warped image using MSER."""
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    H, W = gray.shape[:2]
    centers = []

    for r in regions:
        x, y, w, h = cv2.boundingRect(r.reshape(-1, 1, 2))

        # Reject very small/very large regions
        area = w * h
        if area < 80 or area > (H * W) * 0.02:
            continue

        ar = w / float(h + 1e-6)
        if ar < 0.5 or ar > 2.0:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0
        centers.append((cx, cy, w, h))

    return centers


def filter_ui_regions(centers, H, W):
    """Filter out centers in UI regions (top bar, bottom buttons, etc.)."""
    filtered = []
    for cx, cy, rw, rh in centers:
        if cy < 0.20 * H:       # top 20%
            continue
        if cy > 0.92 * H:       # bottom 8%
            continue
        if cx < 0.08 * W and cy > 0.75 * H:   # bottom-left button zone
            continue
        if cx > 0.85 * W and cy > 0.75 * H:   # bottom-right button zone
            continue
        if cx > 0.85 * W and cy < 0.25 * H:   # top-right badge zone
            continue
        filtered.append((cx, cy, rw, rh))
    return filtered


def dominant_size_cluster(centers, bin_size=4):
    """Keep only spirals that match the dominant size cluster."""
    if not centers:
        return []

    sizes = np.array([0.5*(rw+rh) for (_, _, rw, rh) in centers], dtype=np.float32)
    bins = np.round(sizes / bin_size) * bin_size
    u, c = np.unique(bins, return_counts=True)
    dominant = u[np.argmax(c)]

    kept = []
    for item, b in zip(centers, bins):
        if abs(b - dominant) <= bin_size:
            kept.append(item)
    return kept


def dedupe_centers(centers, dist_thresh=12):
    """Remove duplicate centers that are too close together."""
    kept = []
    for cx, cy, rw, rh in sorted(centers, key=lambda t: (t[1], t[0])):
        ok = True
        for kx, ky, _, _ in kept:
            if np.hypot(cx - kx, cy - ky) < dist_thresh:
                ok = False
                break
        if ok:
            kept.append((cx, cy, rw, rh))
    return kept


def estimate_tile_size(spirals):
    """Estimate tile size from spiral positions using nearest-neighbor distances."""
    if len(spirals) < 2:
        return None

    pts = np.array([(cx, cy) for (cx, cy, _, _) in spirals], dtype=np.float32)

    # Nearest-neighbor distance for each point
    nn = []
    for i in range(len(pts)):
        d = np.sqrt(np.sum((pts - pts[i])**2, axis=1))
        d[i] = 1e9
        nn.append(float(np.min(d)))

    d_nn = float(np.median(nn))

    # Tile size is slightly smaller than center-to-center spacing
    tile = int(round(d_nn * 0.78))
    tile = max(24, tile)

    return tile


def boxes_from_centers(centers, tile_size, H, W):
    """Create bounding boxes from spiral centers."""
    boxes = []
    half = tile_size // 2
    for cx, cy, _, _ in centers:
        x = int(round(cx - half))
        y = int(round(cy - half))
        x = max(0, min(W - tile_size, x))
        y = max(0, min(H - tile_size, y))
        boxes.append((x, y, tile_size, tile_size))
    return boxes


def detect_tiles_via_spirals(warped_bgr):
    """
    Main tile detection pipeline.
    Returns (boxes, spirals, tile_size).
    """
    H, W = warped_bgr.shape[:2]
    spirals = detect_spiral_centers(warped_bgr)

    # 1) Remove obvious UI zones
    spirals = filter_ui_regions(spirals, H, W)

    # 2) Keep dominant spiral size cluster
    spirals = dominant_size_cluster(spirals, bin_size=4)

    # 3) Dedupe near-duplicates
    spirals = dedupe_centers(spirals, dist_thresh=12)

    if len(spirals) < 2:
        return [], spirals, None

    # 4) Estimate tile size
    tile_size = estimate_tile_size(spirals)

    # 5) Build tile boxes
    boxes = boxes_from_centers(spirals, tile_size, H, W)

    return boxes, spirals, tile_size


def tile_edge_score(warped_bgr, box):
    """Compute edge score for a tile (used for flip detection)."""
    x, y, w, h = box
    roi = warped_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 160)
    return float(np.mean(edges))


def tile_flip_score(warped_bgr, box, pad_frac=0.10):
    """
    Combined score using both brightness variance AND edge detection.
    More robust across different tile brightness levels.
    """
    x, y, w, h = box

    # Apply padding to focus on inner tile area
    pad_x = int(w * pad_frac)
    pad_y = int(h * pad_frac)

    x1 = x + pad_x
    y1 = y + pad_y
    x2 = x + w - pad_x
    y2 = y + h - pad_y

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = warped_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    # 1. Brightness variance score
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    variance_score = float(np.std(v))

    # 2. Edge detection score
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 160)
    edge_score = float(np.mean(edges))

    # Combine both metrics
    combined = variance_score + (edge_score * 0.3)

    return combined


def crop_tile_face(warped_bgr, box, pad_frac=0.12):
    """Crop the face area of a tile with padding."""
    x, y, w, h = box
    roi = warped_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return None

    pad = int(min(w, h) * pad_frac)
    if (w - 2*pad) < 10 or (h - 2*pad) < 10:
        return roi
    return roi[pad:h-pad, pad:w-pad]


def init_tile_baseline(warped_bgr, boxes):
    """Initialize baseline flip scores for all tiles."""
    return [tile_flip_score(warped_bgr, b) for b in boxes]

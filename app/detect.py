import cv2
import numpy as np

DEBUG = False

# UI State Machine: CONFIRMING -> SEARCHING -> READY -> RUNNING
UI_STATE = "CONFIRMING"  # "CONFIRMING", "SEARCHING", "READY", "RUNNING"
START_REQUESTED = False
STOP_REQUESTED = False
YES_REQUESTED = False
NO_REQUESTED = False

# Frozen quad - locked iPad position after user confirms
FROZEN_QUAD = None

# Smaller buttons (w=100, h=40 instead of w=160, h=55)
START_BTN = {"x": 20, "y": 0, "w": 100, "h": 40}
STOP_BTN = {"x": 20, "y": 0, "w": 100, "h": 40}
YES_BTN = {"x": 0, "y": 0, "w": 80, "h": 40}
NO_BTN = {"x": 0, "y": 0, "w": 80, "h": 40}

LOCKED_BOXES = None          # list[(x,y,w,h)] once per level
TILE_BACK_BASELINE = None    # list[float] baseline score per tile when back side is showing
TILE_BACK_HASH = {}          # tile_idx -> 64-bit hash of tile's back (spiral) for flip detection
TILE_STATE = None            # list[bool] True if tile is FRONT
FACE_SNAPSHOT = {}   # tile_idx -> BGR image crop (face)
FACE_READY = set()   # tile_idx set for which we've captured face this flip
FACE_HASH = {}          # tile_idx -> 64-bit hash (int)
TILE_ID = {}            # tile_idx -> identity id (int)
ID_COLOR = {}           # identity id -> BGR color
NEXT_ID = {"value": 1}             # identity counter
CAPTURE_DELAY = {}  # tile_idx -> frames remaining before capture


def sanitize_boxes(boxes):
    clean = []
    for (x, y, w, h) in boxes:
        clean.append((int(round(x)), int(round(y)), int(round(w)), int(round(h))))
    return clean

def normalize_face(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))

    # Remove lighting differences
    gray = cv2.equalizeHist(gray)

    # Kill high-frequency autofocus noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray


def tile_edge_score(warped_bgr, box):
    x, y, w, h = box
    roi = warped_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 160)
    return float(np.mean(edges))  # 0..255 (ish)

def tile_flip_score(warped_bgr, box, pad_frac=0.10):
    """
    Combined score using both brightness variance AND edge detection.
    This is more robust across different tile brightness levels.

    pad_frac=0.10 means 10% padding on each side, using ~80% of the box.
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
    # Variance is typically 10-40, edges typically 5-50
    # Scale edge score to contribute roughly equally
    combined = variance_score + (edge_score * 0.3)

    return combined

def crop_tile_face(warped_bgr, box, pad_frac=0.12):
    x, y, w, h = box
    roi = warped_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return None

    pad = int(min(w, h) * pad_frac)
    if (w - 2*pad) < 10 or (h - 2*pad) < 10:
        return roi
    return roi[pad:h-pad, pad:w-pad]

def dhash64(img, hash_size=8):
    """
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
    return (a ^ b).bit_count()

def deterministic_color(idx):
    # nice bright-ish deterministic BGR
    rng = np.random.default_rng(idx)
    c = rng.integers(low=60, high=255, size=3, dtype=np.int32)
    return (int(c[0]), int(c[1]), int(c[2]))  # BGR

def match_identity(new_hash, existing_hash_by_tile, max_dist=10):
    """
    Returns (matched_tile_idx, dist) or (None, None)
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


def init_tile_baseline(warped_bgr, boxes):
    return [tile_flip_score(warped_bgr, b) for b in boxes]


def order_points(pts):
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def angle_cos(p0, p1, p2):
    # cosine of angle at p1 formed by p0-p1-p2 (closer to 0 => closer to 90 degrees)
    v1 = p0 - p1
    v2 = p2 - p1
    return abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))

def quad_diag(pts):
    pts = order_points(pts.astype(np.float32))
    return float(np.linalg.norm(pts[2] - pts[0]))  # tl -> br


def quad_area(pts):
    # polygon area (shoelace)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def iou_like(a, b):
    # cheap "similarity": average corner distance normalized
    return np.mean(np.linalg.norm(a - b, axis=1))

def smooth_quad(prev, new, alpha=0.85):
    # alpha close to 1 => more stable (slower to change)
    return (alpha * prev + (1 - alpha) * new).astype(np.float32)


def four_point_warp(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    # prevent tiny/invalid warps
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

def find_screen(frame):

    h, w = frame.shape[:2]

    # --- BRIGHT SCREEN MASK: prefer the lit screen (works better than bezel on dark mats) ---
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

    # Debug windows (optional but strongly recommended while tuning)
    if DEBUG:
        cv2.imshow("debug mask", mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:40]

    # Debug: show top contour
    dbg = frame.copy()
    cv2.drawContours(dbg, cnts[:1], -1, (0, 255, 255), 3)  # yellow
    if DEBUG:
        cv2.imshow("debug top contour", dbg)

    best = None
    best_score = -1.0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 12000:
            continue

        # minAreaRect fill ratio: contour fills its best-fit rectangle
        rot_rect = cv2.minAreaRect(c)
        (box_w, box_h) = rot_rect[1]
        if box_w < 1 or box_h < 1:
            continue
        fill = float(area) / float(box_w * box_h + 1e-6)
        if fill < 0.70:  # slightly looser because the mask may not be perfect
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

        # --- Angle check: prefer near-90-degree corners ---
        cos1 = angle_cos(tl, tr, br)
        cos2 = angle_cos(tr, br, bl)
        cos3 = angle_cos(br, bl, tl)
        cos4 = angle_cos(bl, tl, tr)
        max_cos = max(cos1, cos2, cos3, cos4)

        # IMPORTANT: 0.15 is too strict; start at 0.35
        if max_cos > 0.35:
            continue
        angle_score = max(0.0, 1.0 - (max_cos / 0.35))

        # width/height from side lengths
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        width = max(widthA, widthB)
        height = max(heightA, heightB)
        if width < 80 or height < 80:
            continue

        # Aspect ratio (normalize >= 1)
        aspect = width / height
        aspect = aspect if aspect >= 1 else 1 / aspect
        if aspect > 2.2:
            continue

        # iPad-ish aspect scoring
        aspect_target = 1.33
        aspect_sigma = 0.28  # a touch looser for real camera skew
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

    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    H, W = gray.shape[:2]
    centers = []

    for r in regions:
        x, y, w, h = cv2.boundingRect(r.reshape(-1, 1, 2))

        # reject very small/very large regions
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
    filtered = []
    for cx, cy, rw, rh in centers:
        if cy < 0.20 * H:       # top 18%
            continue
        if cy > 0.92 * H:       # bottom 8%
            continue
        if cx < 0.08 * W and cy > 0.75 * H:   # bottom-left button zone
            continue
        if cx > 0.85 * W and cy > 0.75 * H:   # bottom-right button zone (if applicable)
            continue
        if cx > 0.85 * W and cy < 0.25 * H:   # top-right badge/pink circle zone
            continue
        filtered.append((cx, cy, rw, rh))
    return filtered

def dominant_size_cluster(centers, bin_size=4):
    import numpy as np

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
    import numpy as np
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
    import numpy as np

    if len(spirals) < 2:
        return None

    pts = np.array([(cx, cy) for (cx, cy, _, _) in spirals], dtype=np.float32)

    # nearest-neighbor distance for each point
    nn = []
    for i in range(len(pts)):
        d = np.sqrt(np.sum((pts - pts[i])**2, axis=1))
        d[i] = 1e9
        nn.append(float(np.min(d)))

    d_nn = float(np.median(nn))

    # Tile size is slightly smaller than center-to-center spacing (spacing includes the gap)
    tile = int(round(d_nn * 0.78))  # tune 0.70–0.90
    tile = max(24, tile)            # clamp lower bound

    return tile

def boxes_from_centers(centers, tile_size, H, W):
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
    import numpy as np

    H, W = warped_bgr.shape[:2]
    spirals = detect_spiral_centers(warped_bgr)  # your MSER-based function

    # 1) remove obvious UI zones
    spirals = filter_ui_regions(spirals, H, W)

    # 2) keep dominant spiral size cluster
    spirals = dominant_size_cluster(spirals, bin_size=4)

    # 3) dedupe near-duplicates
    spirals = dedupe_centers(spirals, dist_thresh=12)

    if len(spirals) < 2:
        return [], spirals, None  # not enough to estimate reliably

    # 4) estimate tile size
    tile_size = estimate_tile_size(spirals)

    # 5) build tile boxes
    boxes = boxes_from_centers(spirals, tile_size, H, W)

    return boxes, spirals, tile_size

def on_mouse(event, x, y, flags, param):
    global START_REQUESTED, STOP_REQUESTED, YES_REQUESTED, NO_REQUESTED, UI_STATE
    if event == cv2.EVENT_LBUTTONDOWN:
        if UI_STATE == "CONFIRMING":
            # Check Yes button
            bx, by, bw, bh = YES_BTN["x"], YES_BTN["y"], YES_BTN["w"], YES_BTN["h"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                YES_REQUESTED = True
            # Check No button
            bx, by, bw, bh = NO_BTN["x"], NO_BTN["y"], NO_BTN["w"], NO_BTN["h"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                NO_REQUESTED = True
        elif UI_STATE == "READY":
            bx, by, bw, bh = START_BTN["x"], START_BTN["y"], START_BTN["w"], START_BTN["h"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                START_REQUESTED = True
        elif UI_STATE == "RUNNING":
            bx, by, bw, bh = STOP_BTN["x"], STOP_BTN["y"], STOP_BTN["w"], STOP_BTN["h"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                STOP_REQUESTED = True


def draw_start_button(img):
    global START_BTN
    H, W = img.shape[:2]
    margin = 15
    START_BTN["x"] = margin
    START_BTN["y"] = H - START_BTN["h"] - margin

    x, y, w, h = START_BTN["x"], START_BTN["y"], START_BTN["w"], START_BTN["h"]

    # Green button background + border
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 80, 40), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 255, 100), 2)

    # Label (centered for smaller button)
    cv2.putText(img, "Start", (x + 18, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)


def draw_stop_button(img):
    global STOP_BTN
    H, W = img.shape[:2]
    margin = 15
    STOP_BTN["x"] = margin
    STOP_BTN["y"] = H - STOP_BTN["h"] - margin

    x, y, w, h = STOP_BTN["x"], STOP_BTN["y"], STOP_BTN["w"], STOP_BTN["h"]

    # Red button background + border
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 80), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 255), 2)

    # Label
    cv2.putText(img, "Stop", (x + 22, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)


def draw_status_message(img, num_tiles):
    """Draw the 'Found X tiles' message in READY state."""
    H, W = img.shape[:2]
    msg = f"Found {num_tiles} tiles. Press Start and begin matching!"

    # Draw background box for readability
    text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (W - text_size[0]) // 2
    text_y = H - 70

    padding = 10
    cv2.rectangle(img,
                  (text_x - padding, text_y - text_size[1] - padding),
                  (text_x + text_size[0] + padding, text_y + padding),
                  (0, 0, 0), -1)
    cv2.rectangle(img,
                  (text_x - padding, text_y - text_size[1] - padding),
                  (text_x + text_size[0] + padding, text_y + padding),
                  (0, 200, 200), 2)

    cv2.putText(img, msg, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def draw_confirm_message(img):
    """Draw the 'Is this correct?' message in CONFIRMING state."""
    H, W = img.shape[:2]
    msg = "iPad detected. Is this correct?"

    text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (W - text_size[0]) // 2
    text_y = H - 70

    padding = 10
    cv2.rectangle(img,
                  (text_x - padding, text_y - text_size[1] - padding),
                  (text_x + text_size[0] + padding, text_y + padding),
                  (0, 0, 0), -1)
    cv2.rectangle(img,
                  (text_x - padding, text_y - text_size[1] - padding),
                  (text_x + text_size[0] + padding, text_y + padding),
                  (200, 200, 0), 2)

    cv2.putText(img, msg, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def draw_yes_no_buttons(img):
    """Draw Yes and No buttons for confirmation."""
    global YES_BTN, NO_BTN
    H, W = img.shape[:2]
    margin = 15
    gap = 20  # gap between buttons

    # Position buttons side by side at bottom left
    YES_BTN["x"] = margin
    YES_BTN["y"] = H - YES_BTN["h"] - margin
    NO_BTN["x"] = margin + YES_BTN["w"] + gap
    NO_BTN["y"] = H - NO_BTN["h"] - margin

    # Draw Yes button (green)
    x, y, w, h = YES_BTN["x"], YES_BTN["y"], YES_BTN["w"], YES_BTN["h"]
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 80, 40), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 255, 100), 2)
    cv2.putText(img, "Yes", (x + 20, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # Draw No button (red)
    x, y, w, h = NO_BTN["x"], NO_BTN["y"], NO_BTN["w"], NO_BTN["h"]
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 80), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 255), 2)
    cv2.putText(img, "No", (x + 24, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)


def draw_dont_move_warning(img):
    """Draw warning message after iPad is locked."""
    H, W = img.shape[:2]
    msg = "iPad locked! Do not move it."

    text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = (W - text_size[0]) // 2
    text_y = 30

    cv2.putText(img, msg, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


if __name__ == "__main__":

    WINDOW_NAME = "MSM Memory Solver"
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    # global LOCKED_BOXES, TILE_BACK_BASELINE, TILE_STATE

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try changing VideoCapture(0) to (1).")

    last_pts = None
    lost_frames = 0
    MAX_LOST = 30  # hold longer for brief glare/reflection drops

    # Placeholder for future solver state
    solver_state = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Default output if we can't produce a warped view yet
        output = np.zeros((360, 640, 3), dtype=np.uint8)
        warped = None  # Will be set if iPad is detected
        cv2.putText(output, "Point camera at iPad...", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Optional debug frame (only shown if DEBUG)
        debug = frame.copy()

        # === STATE: CONFIRMING ===
        # Detect iPad and ask user to confirm before freezing
        if UI_STATE == "CONFIRMING":
            pts = find_screen(frame)
            if pts is not None:
                pts = order_points(pts.astype(np.float32))
                cv2.polylines(debug, [pts.astype(int)], True, (255, 0, 0), 2)

            accepted = False

            if pts is not None:
                area = quad_area(pts)

                if area > 4000:
                    if last_pts is None:
                        last_pts = pts
                        accepted = True
                    else:
                        last_ord = order_points(last_pts.astype(np.float32))
                        d = iou_like(last_ord, pts)
                        diag = np.linalg.norm(last_ord[2] - last_ord[0])
                        jump_thresh = max(60.0, 0.08 * diag)

                        if d < jump_thresh:
                            new_diag = quad_diag(pts)
                            old_diag = quad_diag(last_ord)
                            scale = new_diag / (old_diag + 1e-6)

                            if 0.92 <= scale <= 1.08:
                                last_pts = smooth_quad(last_ord, pts, alpha=0.85)
                                accepted = True

            if accepted:
                lost_frames = 0
            else:
                lost_frames += 1
                if lost_frames > MAX_LOST:
                    last_pts = None

            # Show confirmation UI if iPad detected
            if last_pts is not None:
                last_ord = order_points(last_pts.astype(np.float32))
                cv2.polylines(debug, [last_ord.astype(int)], True, (0, 255, 0), 3)

                warped = four_point_warp(frame, last_ord)
                if warped is not None:
                    output = warped.copy()
                    cv2.putText(output, "CONFIRMING - Check iPad position",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    draw_confirm_message(output)
                    draw_yes_no_buttons(output)

        # === STATES: SEARCHING, READY, RUNNING ===
        # Use frozen quad - no more detection jitter
        elif FROZEN_QUAD is not None:
            warped = four_point_warp(frame, FROZEN_QUAD)
            if warped is not None:
                output = warped.copy()

                # Draw "iPad locked" warning at top
                draw_dont_move_warning(output)

                # === STATE: SEARCHING ===
                # Look for tiles, transition to READY when found
                if UI_STATE == "SEARCHING":
                    if LOCKED_BOXES is None:
                        boxes, spirals, tile_size = detect_tiles_via_spirals(warped)

                        # Only lock if it looks like a real level (at least 2 tiles)
                        if len(boxes) >= 2:
                            LOCKED_BOXES = sanitize_boxes(boxes)

                    # Draw detected tiles (preview)
                    if LOCKED_BOXES is not None:
                        for i, b in enumerate(LOCKED_BOXES):
                            x, y, w, h = map(int, map(round, b))
                            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        cv2.putText(output, f"tiles={len(LOCKED_BOXES)}  (detecting...)",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                        # Transition to READY
                        UI_STATE = "READY"
                    else:
                        cv2.putText(output, "Looking for tiles (show spirals)...",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # === STATE: READY ===
                # Show tiles and wait for Start button
                elif UI_STATE == "READY":
                    # Draw tile boxes (preview only, no tracking)
                    if LOCKED_BOXES is not None:
                        for i, b in enumerate(LOCKED_BOXES):
                            x, y, w, h = map(int, map(round, b))
                            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        cv2.putText(output, f"tiles={len(LOCKED_BOXES)}  READY",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                        # Draw status message and Start button
                        draw_status_message(output, len(LOCKED_BOXES))
                        draw_start_button(output)

                # === STATE: RUNNING ===
                # Active tile tracking and matching
                elif UI_STATE == "RUNNING":
                    # Hash-based flip detection thresholds
                    # Hash distance > FLIP_HASH_THRESHOLD means tile looks different (flipped)
                    FLIP_HASH_THRESHOLD = 12
                    # Hash distance < BACK_HASH_THRESHOLD means tile looks like original (back)
                    BACK_HASH_THRESHOLD = 8

                    max_hash_dist = 0
                    max_i = -1

                    # Wait ~0.4s for flip animation (0.29s) + finger to clear
                    # At 30fps: 12 frames = 0.4s
                    DELAY_FRAMES = 12

                    for i, b in enumerate(LOCKED_BOXES):
                        x, y, w, h = map(int, map(round, b))

                        # If tile has an identity, draw match number
                        if i in TILE_ID:
                            tid = TILE_ID[i]
                            # Draw match number prominently in center
                            label = str(tid)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.5
                            thickness = 3
                            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                            text_x = x + (w - text_size[0]) // 2
                            text_y = y + (h + text_size[1]) // 2
                            # Black outline for visibility
                            cv2.putText(output, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
                            # White number
                            cv2.putText(output, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

                        # Draw tile box for all tiles
                        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Skip flip detection for already-captured tiles
                        if i in FACE_HASH:
                            continue

                        # Compute current hash and compare to baseline
                        tile_img = crop_tile_face(warped, b)
                        if tile_img is None:
                            continue
                        norm = normalize_face(tile_img)
                        current_hash = dhash64(norm)

                        # Get baseline hash (spiral back)
                        if i not in TILE_BACK_HASH:
                            continue
                        baseline_hash = TILE_BACK_HASH[i]

                        # Hash distance: higher = more different from baseline
                        hash_dist = hamming64(current_hash, baseline_hash)

                        # Track max hash distance for debug display
                        if hash_dist > max_hash_dist:
                            max_hash_dist = hash_dist
                            max_i = i

                        # --- BACK -> FRONT transition ---
                        if TILE_STATE[i] is False and hash_dist > FLIP_HASH_THRESHOLD:
                            TILE_STATE[i] = True
                            CAPTURE_DELAY[i] = DELAY_FRAMES

                        # --- FRONT -> BACK transition ---
                        if TILE_STATE[i] is True:
                            if i in FACE_READY:
                                if hash_dist < BACK_HASH_THRESHOLD:
                                    TILE_STATE[i] = False
                                    FACE_READY.discard(i)
                                    CAPTURE_DELAY[i] = 0

                        # --- If tile is FRONT, count down then capture ---
                        if TILE_STATE[i] is True:
                            if CAPTURE_DELAY.get(i, 0) > 0:
                                CAPTURE_DELAY[i] -= 1
                            else:
                                # Confirm tile still looks different (not a false trigger)
                                if hash_dist < FLIP_HASH_THRESHOLD:
                                    # Hash distance dropped - was probably finger or noise
                                    TILE_STATE[i] = False
                                    CAPTURE_DELAY[i] = 0
                                    continue

                                face = crop_tile_face(warped, b)
                                if face is not None:
                                    FACE_SNAPSHOT[i] = face
                                    FACE_READY.add(i)

                                    norm = normalize_face(face)
                                    h = dhash64(norm)

                                    FACE_HASH[i] = h

                                    existing = {k: v for k, v in FACE_HASH.items() if k != i}
                                    matched_tile, dist = match_identity(h, existing, max_dist=15)

                                    if matched_tile is not None and matched_tile in TILE_ID:
                                        TILE_ID[i] = TILE_ID[matched_tile]
                                    else:
                                        tid = NEXT_ID["value"]
                                        NEXT_ID["value"] += 1
                                        TILE_ID[i] = tid
                                        ID_COLOR[tid] = deterministic_color(tid)

                        # If flipped, draw the tile index label
                        if TILE_STATE[i]:
                            cv2.putText(output, f"{i}", (x + 10, y + 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                            if i in FACE_SNAPSHOT:
                                cv2.putText(output, "S", (x + w - 25, y + 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

                    cv2.putText(output, f"saved={len(FACE_SNAPSHOT)}",
                                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    cv2.putText(output, f"maxHashDist={max_hash_dist} tile={max_i} (threshold={FLIP_HASH_THRESHOLD})",
                                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.putText(output, f"tiles={len(LOCKED_BOXES)}  RUNNING",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                    # Draw Stop button
                    draw_stop_button(output)

        # === Handle Yes button click (confirm iPad position) ===
        if YES_REQUESTED:
            YES_REQUESTED = False
            if UI_STATE == "CONFIRMING" and last_pts is not None:
                # Freeze the quad - no more detection updates
                FROZEN_QUAD = order_points(last_pts.astype(np.float32))
                UI_STATE = "SEARCHING"

        # === Handle No button click (re-detect iPad) ===
        if NO_REQUESTED:
            NO_REQUESTED = False
            if UI_STATE == "CONFIRMING":
                # Reset detection, stay in CONFIRMING
                last_pts = None
                lost_frames = 0

        # === Handle Start button click ===
        if START_REQUESTED:
            START_REQUESTED = False
            if UI_STATE == "READY" and LOCKED_BOXES is not None and warped is not None:
                # Initialize tracking state when starting
                TILE_BACK_BASELINE = init_tile_baseline(warped, LOCKED_BOXES)
                TILE_STATE = [False] * len(LOCKED_BOXES)
                CAPTURE_DELAY.clear()
                CAPTURE_DELAY.update({i: 0 for i in range(len(LOCKED_BOXES))})

                # Compute baseline hash for each tile (spiral back)
                TILE_BACK_HASH.clear()
                for i, box in enumerate(LOCKED_BOXES):
                    tile_img = crop_tile_face(warped, box)
                    if tile_img is not None:
                        norm = normalize_face(tile_img)
                        TILE_BACK_HASH[i] = dhash64(norm)

                UI_STATE = "RUNNING"

        # === Handle Stop button click ===
        if STOP_REQUESTED:
            STOP_REQUESTED = False

            # Clear all state
            solver_state.clear()
            LOCKED_BOXES = None
            TILE_BACK_BASELINE = None
            TILE_BACK_HASH.clear()
            TILE_STATE = None
            FACE_SNAPSHOT.clear()
            FACE_READY.clear()
            FACE_HASH.clear()
            TILE_ID.clear()
            ID_COLOR.clear()
            NEXT_ID["value"] = 1
            CAPTURE_DELAY.clear()

            # Clear frozen quad and go back to CONFIRMING
            FROZEN_QUAD = None
            last_pts = None
            lost_frames = 0
            UI_STATE = "CONFIRMING"

        # Show exactly one window
        cv2.imshow(WINDOW_NAME, output)

        # Optional debug window(s)
        if DEBUG:
            cv2.imshow("debug (detected outline)", debug)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

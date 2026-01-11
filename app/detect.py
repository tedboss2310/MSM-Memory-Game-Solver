import cv2
import numpy as np

DEBUG = False
RESET_REQUESTED = False
RESET_BTN = {"x": 20, "y": 0, "w": 160, "h": 55}  # y will be set once we know image height

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
        if cy < 0.18 * H:       # top 18%
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
    global RESET_REQUESTED, RESET_BTN
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh = RESET_BTN["x"], RESET_BTN["y"], RESET_BTN["w"], RESET_BTN["h"]
        if bx <= x <= bx + bw and by <= y <= by + bh:
            RESET_REQUESTED = True

def draw_reset_button(img):
    global RESET_BTN
    H, W = img.shape[:2]
    margin = 20
    RESET_BTN["x"] = margin
    RESET_BTN["y"] = H - RESET_BTN["h"] - margin

    x, y, w, h = RESET_BTN["x"], RESET_BTN["y"], RESET_BTN["w"], RESET_BTN["h"]

    # Button background + border
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (240, 240, 240), 2)

    # Label
    cv2.putText(img, "Reset", (x + 35, y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)


if __name__ == "__main__":

    WINDOW_NAME = "MSM Memory Solver"
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

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
        cv2.putText(output, "Point camera at iPad...", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Optional debug frame (only shown if DEBUG)
        debug = frame.copy()

        pts = find_screen(frame)
        if pts is not None:
            pts = order_points(pts.astype(np.float32))
            cv2.polylines(debug, [pts.astype(int)], True, (255, 0, 0), 2)  # blue = raw detection

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

        # If we have a stable iPad quad, generate the main output from warped view
        if last_pts is not None:
            last_ord = order_points(last_pts.astype(np.float32))
            cv2.polylines(debug, [last_ord.astype(int)], True, (0, 255, 0), 3)  # green tracking quad

            warped = four_point_warp(frame, last_ord)
            if warped is not None:
                boxes, spirals, tile_size = detect_tiles_via_spirals(warped)

                output = warped.copy()

                # Draw tile boxes (green). In normal mode, don't draw spiral dots.
                for (x, y, w, h) in boxes:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # If you want spiral dots only in DEBUG:
                if DEBUG:
                    for cx, cy, rw, rh in spirals:
                        cv2.circle(output, (int(cx), int(cy)), 6, (0, 255, 255), -1)

                # Small HUD
                cv2.putText(output, f"tiles={len(boxes)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # Draw the Reset button (bottom-left)
                draw_reset_button(output)

        # Handle reset click (keep the callback dumb; do the work here)
        if RESET_REQUESTED:
            RESET_REQUESTED = False

            # Clear future solver state
            solver_state.clear()

            # (Optional) also drop tracking so it reacquires iPad
            # last_pts = None
            # lost_frames = 0

            cv2.putText(output, "RESET!", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

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

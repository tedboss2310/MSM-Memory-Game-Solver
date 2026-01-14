"""
MSM Memory Solver - Main entry point and game loop.
"""

import cv2
import numpy as np

import config
from geometry import order_points, quad_area, quad_diag, iou_like, smooth_quad, four_point_warp, sanitize_boxes
from detection import find_screen, detect_tiles_via_spirals, crop_tile_face, init_tile_baseline
from hashing import normalize_face, dhash64, hamming64, match_identity, deterministic_color
from ui import (on_mouse, draw_start_button, draw_stop_button, draw_status_message,
                draw_confirm_message, draw_yes_no_buttons, draw_dont_move_warning)


def main():
    WINDOW_NAME = "MSM Memory Solver"
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try changing VideoCapture(0) to (1).")

    last_pts = None
    lost_frames = 0

    # Placeholder for future solver state
    solver_state = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Default output if we can't produce a warped view yet
        output = np.zeros((360, 640, 3), dtype=np.uint8)
        warped = None
        cv2.putText(output, "Point camera at iPad...", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Optional debug frame
        debug = frame.copy()

        # === STATE: CONFIRMING ===
        if config.UI_STATE == "CONFIRMING":
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
                if lost_frames > config.MAX_LOST:
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
        elif config.FROZEN_QUAD is not None:
            warped = four_point_warp(frame, config.FROZEN_QUAD)
            if warped is not None:
                output = warped.copy()

                # Draw "iPad locked" warning at top
                draw_dont_move_warning(output)

                # === STATE: SEARCHING ===
                if config.UI_STATE == "SEARCHING":
                    if config.LOCKED_BOXES is None:
                        boxes, spirals, tile_size = detect_tiles_via_spirals(warped)

                        # Only lock if it looks like a real level (at least 2 tiles)
                        if len(boxes) >= 2:
                            config.LOCKED_BOXES = sanitize_boxes(boxes)

                    # Draw detected tiles (preview)
                    if config.LOCKED_BOXES is not None:
                        for i, b in enumerate(config.LOCKED_BOXES):
                            x, y, w, h = map(int, map(round, b))
                            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        cv2.putText(output, f"tiles={len(config.LOCKED_BOXES)}  (detecting...)",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                        # Transition to READY
                        config.UI_STATE = "READY"
                    else:
                        cv2.putText(output, "Looking for tiles (show spirals)...",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # === STATE: READY ===
                elif config.UI_STATE == "READY":
                    # Draw tile boxes (preview only, no tracking)
                    if config.LOCKED_BOXES is not None:
                        for i, b in enumerate(config.LOCKED_BOXES):
                            x, y, w, h = map(int, map(round, b))
                            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        cv2.putText(output, f"tiles={len(config.LOCKED_BOXES)}  READY",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                        # Draw status message and Start button
                        draw_status_message(output, len(config.LOCKED_BOXES))
                        draw_start_button(output)

                # === STATE: RUNNING ===
                elif config.UI_STATE == "RUNNING":
                    max_hash_dist = 0
                    max_i = -1

                    for i, b in enumerate(config.LOCKED_BOXES):
                        x, y, w, h = map(int, map(round, b))

                        # If tile has an identity, draw match number
                        if i in config.TILE_ID:
                            tid = config.TILE_ID[i]
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
                        if i in config.FACE_HASH:
                            continue

                        # Compute current hash and compare to baseline
                        tile_img = crop_tile_face(warped, b)
                        if tile_img is None:
                            continue
                        norm = normalize_face(tile_img)
                        current_hash = dhash64(norm)

                        # Get baseline hash (spiral back)
                        if i not in config.TILE_BACK_HASH:
                            continue
                        baseline_hash = config.TILE_BACK_HASH[i]

                        # Hash distance: higher = more different from baseline
                        hash_dist = hamming64(current_hash, baseline_hash)

                        # Track max hash distance for debug display
                        if hash_dist > max_hash_dist:
                            max_hash_dist = hash_dist
                            max_i = i

                        # --- BACK -> FRONT transition (with hysteresis) ---
                        if config.TILE_STATE[i] is False:
                            if hash_dist > config.FLIP_HASH_THRESHOLD:
                                # Increment confirmation counter
                                config.FLIP_CONFIRM_COUNT[i] = config.FLIP_CONFIRM_COUNT.get(i, 0) + 1
                                # Only transition after enough consecutive frames
                                if config.FLIP_CONFIRM_COUNT[i] >= config.FLIP_CONFIRM_FRAMES:
                                    config.TILE_STATE[i] = True
                                    config.CAPTURE_DELAY[i] = config.DELAY_FRAMES
                                    config.FLIP_CONFIRM_COUNT[i] = 0
                            else:
                                # Reset counter if below threshold
                                config.FLIP_CONFIRM_COUNT[i] = 0

                        # --- FRONT -> BACK transition ---
                        if config.TILE_STATE[i] is True:
                            if i in config.FACE_READY:
                                if hash_dist < config.BACK_HASH_THRESHOLD:
                                    config.TILE_STATE[i] = False
                                    config.FACE_READY.discard(i)
                                    config.CAPTURE_DELAY[i] = 0

                        # --- If tile is FRONT, count down then capture ---
                        if config.TILE_STATE[i] is True:
                            if config.CAPTURE_DELAY.get(i, 0) > 0:
                                config.CAPTURE_DELAY[i] -= 1
                            else:
                                # Confirm tile still looks different (not a false trigger)
                                if hash_dist < config.FLIP_HASH_THRESHOLD:
                                    config.TILE_STATE[i] = False
                                    config.CAPTURE_DELAY[i] = 0
                                    continue

                                face = crop_tile_face(warped, b)
                                if face is not None:
                                    config.FACE_SNAPSHOT[i] = face
                                    config.FACE_READY.add(i)

                                    norm = normalize_face(face)
                                    h = dhash64(norm)

                                    config.FACE_HASH[i] = h

                                    existing = {k: v for k, v in config.FACE_HASH.items() if k != i}
                                    matched_tile, dist = match_identity(h, existing, max_dist=config.MATCH_MAX_DIST)

                                    if matched_tile is not None and matched_tile in config.TILE_ID:
                                        config.TILE_ID[i] = config.TILE_ID[matched_tile]
                                    else:
                                        tid = config.NEXT_ID["value"]
                                        config.NEXT_ID["value"] += 1
                                        config.TILE_ID[i] = tid
                                        config.ID_COLOR[tid] = deterministic_color(tid)

                        # If flipped, draw the tile index label
                        if config.TILE_STATE[i]:
                            cv2.putText(output, f"{i}", (x + 10, y + 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                            if i in config.FACE_SNAPSHOT:
                                cv2.putText(output, "S", (x + w - 25, y + 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

                    cv2.putText(output, f"saved={len(config.FACE_SNAPSHOT)}",
                                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    cv2.putText(output, f"maxHashDist={max_hash_dist} tile={max_i} (threshold={config.FLIP_HASH_THRESHOLD})",
                                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.putText(output, f"tiles={len(config.LOCKED_BOXES)}  RUNNING",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                    # Draw Stop button
                    draw_stop_button(output)

        # === Handle Yes button click (confirm iPad position) ===
        if config.YES_REQUESTED:
            config.YES_REQUESTED = False
            if config.UI_STATE == "CONFIRMING" and last_pts is not None:
                config.FROZEN_QUAD = order_points(last_pts.astype(np.float32))
                config.UI_STATE = "SEARCHING"

        # === Handle No button click (re-detect iPad) ===
        if config.NO_REQUESTED:
            config.NO_REQUESTED = False
            if config.UI_STATE == "CONFIRMING":
                last_pts = None
                lost_frames = 0

        # === Handle Start button click ===
        if config.START_REQUESTED:
            config.START_REQUESTED = False
            if config.UI_STATE == "READY" and config.LOCKED_BOXES is not None and warped is not None:
                # Initialize tracking state when starting
                config.TILE_BACK_BASELINE = init_tile_baseline(warped, config.LOCKED_BOXES)
                config.TILE_STATE = [False] * len(config.LOCKED_BOXES)
                config.CAPTURE_DELAY.clear()
                config.CAPTURE_DELAY.update({i: 0 for i in range(len(config.LOCKED_BOXES))})

                # Compute baseline hash for each tile (spiral back)
                config.TILE_BACK_HASH.clear()
                for i, box in enumerate(config.LOCKED_BOXES):
                    tile_img = crop_tile_face(warped, box)
                    if tile_img is not None:
                        norm = normalize_face(tile_img)
                        config.TILE_BACK_HASH[i] = dhash64(norm)

                config.UI_STATE = "RUNNING"

        # === Handle Stop button click ===
        if config.STOP_REQUESTED:
            config.STOP_REQUESTED = False

            # Clear all state
            solver_state.clear()
            config.LOCKED_BOXES = None
            config.TILE_BACK_BASELINE = None
            config.TILE_BACK_HASH.clear()
            config.TILE_STATE = None
            config.FACE_SNAPSHOT.clear()
            config.FACE_READY.clear()
            config.FACE_HASH.clear()
            config.TILE_ID.clear()
            config.ID_COLOR.clear()
            config.NEXT_ID["value"] = 1
            config.CAPTURE_DELAY.clear()
            config.FLIP_CONFIRM_COUNT.clear()

            # Clear frozen quad and go back to CONFIRMING
            config.FROZEN_QUAD = None
            last_pts = None
            lost_frames = 0
            config.UI_STATE = "CONFIRMING"

        # Show exactly one window
        cv2.imshow(WINDOW_NAME, output)

        # Optional debug window(s)
        if config.DEBUG:
            cv2.imshow("debug (detected outline)", debug)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

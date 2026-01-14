"""
UI drawing functions for buttons, messages, and overlays.
"""

import cv2
import config


def on_mouse(event, x, y, flags, param):
    """Mouse callback handler for button clicks."""
    if event == cv2.EVENT_LBUTTONDOWN:
        if config.UI_STATE == "CONFIRMING":
            # Check Yes button
            bx, by, bw, bh = config.YES_BTN["x"], config.YES_BTN["y"], config.YES_BTN["w"], config.YES_BTN["h"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                config.YES_REQUESTED = True
            # Check No button
            bx, by, bw, bh = config.NO_BTN["x"], config.NO_BTN["y"], config.NO_BTN["w"], config.NO_BTN["h"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                config.NO_REQUESTED = True
        elif config.UI_STATE == "READY":
            bx, by, bw, bh = config.START_BTN["x"], config.START_BTN["y"], config.START_BTN["w"], config.START_BTN["h"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                config.START_REQUESTED = True
        elif config.UI_STATE == "RUNNING":
            bx, by, bw, bh = config.STOP_BTN["x"], config.STOP_BTN["y"], config.STOP_BTN["w"], config.STOP_BTN["h"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                config.STOP_REQUESTED = True


def draw_start_button(img):
    """Draw Start button in READY state."""
    H, W = img.shape[:2]
    margin = 15
    config.START_BTN["x"] = margin
    config.START_BTN["y"] = H - config.START_BTN["h"] - margin

    x, y, w, h = config.START_BTN["x"], config.START_BTN["y"], config.START_BTN["w"], config.START_BTN["h"]

    # Green button background + border
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 80, 40), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 255, 100), 2)

    # Label
    cv2.putText(img, "Start", (x + 18, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)


def draw_stop_button(img):
    """Draw Stop button in RUNNING state."""
    H, W = img.shape[:2]
    margin = 15
    config.STOP_BTN["x"] = margin
    config.STOP_BTN["y"] = H - config.STOP_BTN["h"] - margin

    x, y, w, h = config.STOP_BTN["x"], config.STOP_BTN["y"], config.STOP_BTN["w"], config.STOP_BTN["h"]

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
    H, W = img.shape[:2]
    margin = 15
    gap = 20

    # Position buttons side by side at bottom left
    config.YES_BTN["x"] = margin
    config.YES_BTN["y"] = H - config.YES_BTN["h"] - margin
    config.NO_BTN["x"] = margin + config.YES_BTN["w"] + gap
    config.NO_BTN["y"] = H - config.NO_BTN["h"] - margin

    # Draw Yes button (green)
    x, y, w, h = config.YES_BTN["x"], config.YES_BTN["y"], config.YES_BTN["w"], config.YES_BTN["h"]
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 80, 40), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 255, 100), 2)
    cv2.putText(img, "Yes", (x + 20, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # Draw No button (red)
    x, y, w, h = config.NO_BTN["x"], config.NO_BTN["y"], config.NO_BTN["w"], config.NO_BTN["h"]
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

"""
Global state variables and constants for MSM Memory Solver.
"""

DEBUG = False

# UI State Machine: CONFIRMING -> SEARCHING -> READY -> RUNNING
UI_STATE = "CONFIRMING"  # "CONFIRMING", "SEARCHING", "READY", "RUNNING"
START_REQUESTED = False
STOP_REQUESTED = False
YES_REQUESTED = False
NO_REQUESTED = False

# Frozen quad - locked iPad position after user confirms
FROZEN_QUAD = None

# Button dimensions and positions
START_BTN = {"x": 20, "y": 0, "w": 100, "h": 40}
STOP_BTN = {"x": 20, "y": 0, "w": 100, "h": 40}
YES_BTN = {"x": 0, "y": 0, "w": 80, "h": 40}
NO_BTN = {"x": 0, "y": 0, "w": 80, "h": 40}

# Tile detection state
LOCKED_BOXES = None          # list[(x,y,w,h)] once per level
TILE_BACK_BASELINE = None    # list[float] baseline score per tile when back side is showing
TILE_BACK_HASH = {}          # tile_idx -> 64-bit hash of tile's back (spiral) for flip detection
TILE_STATE = None            # list[bool] True if tile is FRONT

# Face capture and matching state
FACE_SNAPSHOT = {}   # tile_idx -> BGR image crop (face)
FACE_READY = set()   # tile_idx set for which we've captured face this flip
FACE_HASH = {}       # tile_idx -> 64-bit hash (int)
TILE_ID = {}         # tile_idx -> identity id (int)
ID_COLOR = {}        # identity id -> BGR color
NEXT_ID = {"value": 1}  # identity counter
CAPTURE_DELAY = {}   # tile_idx -> frames remaining before capture
FLIP_CONFIRM_COUNT = {}  # tile_idx -> consecutive frames above threshold (for hysteresis)

# Hash-based flip detection thresholds
FLIP_HASH_THRESHOLD = 14   # Hash distance > this means tile looks different (flipped) - raised from 12
BACK_HASH_THRESHOLD = 8    # Hash distance < this means tile looks like original (back)
FLIP_CONFIRM_FRAMES = 5    # Require this many consecutive frames above threshold to confirm flip

# Face matching threshold (lower = stricter, must be more similar to match)
MATCH_MAX_DIST = 10        # Max hamming distance to consider two faces a match (was 15)

# Timing constants
DELAY_FRAMES = 12  # Wait ~0.4s for flip animation (0.29s) + finger to clear (at 30fps)
MAX_LOST = 30      # Hold detection longer for brief glare/reflection drops

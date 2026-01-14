# My Singing Monsters Memory Game Solver

![Alt text](/MSM-memory-demo.png "Demo")


A computer vision-based tool that helps you solve the Memory Game mini-game in My Singing Monsters. Point your webcam at your device, and the solver will track which tiles you've flipped and identify matching pairs.

## Overview

The Memory Game in My Singing Monsters presents a grid of face-down tiles. When you tap a tile, it flips to reveal a monster face. Your goal is to find matching pairs by remembering which faces are behind which tiles.

This solver uses your webcam to:
1. Detect your iPad/tablet screen
2. Identify the tile grid
3. Track when you flip tiles
4. Recognize and remember each monster face using perceptual hashing
5. Display match numbers so you know which tiles are pairs

The solver doesn't play the game for you - it acts as a memory aid, showing you which tiles match as you discover them.

## Requirements

- Python 3.8+
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- A webcam pointed at your device

## Installation

```bash
git clone https://github.com/tedboss2310/MSM-Memory-Solver.git
cd MSM-Memory-Solver
pip install opencv-python numpy
```

## Usage

1. **Start the solver:**
   ```bash
   cd app
   python3 main.py
   ```

2. **Position your camera:**
   - Point your webcam at your iPad/tablet
   - Make sure the entire screen is visible
   - Avoid glare and reflections

3. **Confirm iPad detection:**
   - The solver will detect your screen and draw a green outline
   - Click **Yes** if the detection looks correct, or **No** to re-detect
   - Once confirmed, the iPad position is locked (don't move it!)

4. **Wait for tile detection:**
   - Navigate to the Memory Game in My Singing Monsters
   - The solver will detect the tile grid using the spiral patterns on tile backs
   - You'll see green boxes around each detected tile

5. **Start tracking:**
   - Click **Start** when you're ready to begin
   - The solver captures a baseline image of each tile's back

6. **Play the game:**
   - Tap tiles on your device as normal
   - When a tile flips, the solver detects it and captures the face
   - Matching faces are assigned the same number
   - Look for tiles with the same number - those are your pairs!

7. **Reset for next level:**
   - Click **Stop** to reset all state
   - The solver returns to iPad confirmation mode

**Keyboard shortcuts:**
- `Q` - Quit the application

## How It Works

### Screen Detection
The solver finds your iPad screen using brightness-based segmentation and contour detection. It looks for a bright rectangular region with approximately 4:3 aspect ratio and near-90-degree corners. Once confirmed, the screen position is "frozen" to eliminate detection jitter.

### Tile Detection
Tiles are detected by finding the spiral patterns on their backs using MSER (Maximally Stable Extremal Regions). The solver:
1. Detects spiral-like blob centers
2. Filters out UI regions (top bar, bottom buttons)
3. Clusters spirals by size to find the dominant tile size
4. Removes duplicates
5. Estimates tile dimensions from nearest-neighbor distances

### Flip Detection
The solver uses **perceptual hashing** (dHash) to detect when tiles flip:
1. At start, it captures a baseline hash of each tile's back (spiral)
2. Each frame, it computes the current hash and compares to baseline
3. If the Hamming distance exceeds the threshold for several consecutive frames, the tile is considered flipped
4. This approach is robust to lighting changes and works regardless of face brightness

### Face Matching
When a tile is flipped:
1. The solver waits for the flip animation to complete
2. Captures and normalizes the face image (grayscale, histogram equalization, blur)
3. Computes a 64-bit dHash of the face
4. Compares against all previously seen faces using Hamming distance
5. If a match is found within the threshold, assigns the same ID number

### State Machine
The UI follows a state machine:
```
CONFIRMING → SEARCHING → READY → RUNNING
     ↑                              |
     └──────────── (Stop) ──────────┘
```

## Configuration

All tunable parameters are in `app/config.py`:

### Flip Detection

| Variable | Default | Description |
|----------|---------|-------------|
| `FLIP_HASH_THRESHOLD` | 14 | Hamming distance above this = tile looks different (flipped). Increase if getting false flips. |
| `BACK_HASH_THRESHOLD` | 8 | Hamming distance below this = tile looks like original (back). |
| `FLIP_CONFIRM_FRAMES` | 5 | Consecutive frames above threshold required to confirm a flip. Increase if getting false triggers from noise. |

### Face Matching

| Variable | Default | Description |
|----------|---------|-------------|
| `MATCH_MAX_DIST` | 10 | Max Hamming distance to consider two faces a match. Lower = stricter matching. Range: 0 (identical) to 64 (completely different). |

### Timing

| Variable | Default | Description |
|----------|---------|-------------|
| `DELAY_FRAMES` | 12 | Frames to wait after flip detection before capturing face (~0.4s at 30fps). Allows flip animation to complete. |
| `MAX_LOST` | 30 | Frames to hold iPad detection when temporarily lost (glare/reflection). |

### Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | False | Set to `True` to show debug windows (mask visualization, contour detection). |

## Troubleshooting

**Tiles are being detected as flipped when I haven't tapped them:**
- Increase `FLIP_HASH_THRESHOLD` (try 16-18)
- Increase `FLIP_CONFIRM_FRAMES` (try 7-10)
- Make sure your camera is stable and not auto-focusing constantly

**Matching tiles aren't being recognized as matches:**
- Increase `MATCH_MAX_DIST` (try 12-15)
- Make sure lighting is consistent across the screen

**Tiles aren't being detected at all:**
- Make sure the spiral backs are visible
- Check that you're in the Memory Game (not another screen)
- Try adjusting camera angle to reduce glare

**iPad not being detected:**
- Increase screen brightness
- Reduce ambient light / glare
- Make sure the full screen is in frame

## Project Structure

```
app/
├── main.py        # Entry point and main game loop
├── config.py      # Global state variables and tunable constants
├── detection.py   # Screen and tile detection functions
├── hashing.py     # Perceptual hashing and face matching
├── geometry.py    # Geometric helpers (perspective transform, etc.)
└── ui.py          # UI drawing functions (buttons, messages)
```

## License

MIT License

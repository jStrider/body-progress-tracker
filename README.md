# body-align

A CLI tool that automatically aligns progress photos using MediaPipe pose detection and rembg background removal. Given a set of front, back, and profile photos, it detects body landmarks, corrects rotation, normalizes scale so the full body consistently fills the frame, mirrors profiles to face right, removes the background, and outputs clean 800×1067 images — making it trivial to compare photos across different dates.

## Install

```bash
# From PyPI (once published)
pip install body-align

# From source
git clone https://github.com/jStrider/body-progress-tracker.git
cd body-progress-tracker
pip install .
```

> **Note:** First run downloads ~500MB of ML models (MediaPipe pose + segmentation) to `~/.cache/body-align/`. This is a one-time download.

## Requirements

- Python >= 3.10
- ~500MB disk space for auto-downloaded models

## Usage

```bash
# Positional: front back profile
body-align front.jpg back.jpg side.jpg

# Named flags
body-align --front front.jpg --back back.jpg --profile side.jpg

# Custom output directory and date label
body-align front.jpg back.jpg side.jpg --output ./results --date 2026-03-28

# Custom background color and size
body-align front.jpg back.jpg side.jpg --bg-color "#000000" --size 1080x1440

# Skip background removal (faster)
body-align front.jpg back.jpg side.jpg --no-bg
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output DIR` | `./aligned` | Output directory |
| `--date YYYY-MM-DD` | today | Label for output subdirectory |
| `--bg-color HEX` | `#1E1E1E` | Background color after removal |
| `--size WxH` | `800x1067` | Output image dimensions |
| `--no-bg` | off | Skip background removal (faster) |

Output is saved to `<output>/<date>/front.jpg`, `back.jpg`, `profile.jpg`.

## How It Works

1. **Pose detection** — MediaPipe Pose Landmarker detects body keypoints (shoulders, hips, ankles, face)
2. **Rotation correction** — Rotates image to level the shoulders + correct overall body lean
3. **Scale normalization** — Scales so the head-to-feet distance is consistent across all photos (fills ~94% of frame height)
4. **Centering** — Horizontally centers on the shoulder midpoint
5. **Profile mirroring** — Ensures profile shots always face right
6. **Force-align pass** — MediaPipe segmentation finds the exact person bounding box for pixel-perfect positioning
7. **Background removal** — rembg removes the background and replaces it with a solid color

## Python API

```python
from body_align import align_photos

result = align_photos(
    front_path="front.jpg",
    back_path="back.jpg",
    profile_path="side.jpg",
    output_dir="./aligned",
    date="2026-03-28",
    options={
        "bg_color": "#1E1E1E",
        "size": "800x1067",
        "no_bg": False,
    }
)

print(result)
# {
#   "front": "./aligned/2026-03-28/front.jpg",
#   "back": "./aligned/2026-03-28/back.jpg",
#   "profile": "./aligned/2026-03-28/profile.jpg",
#   "success": True,
#   "date": "2026-03-28"
# }
```

## License

MIT

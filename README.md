# Body Progress Tracker

Self-hosted body transformation tracker with **automatic photo alignment** using MediaPipe pose detection + rembg background removal.

> Take a photo every day → see your physique evolution with perfectly aligned, normalized images.

![Example comparison](docs/example.png)

## Features

- **Automatic alignment** — levels shoulders, corrects body tilt, normalizes height
- **Pixel-perfect normalization** — head and feet always at the same position across all photos
- **Background removal** — uniform dark background via rembg (no distracting environments)
- **Profile detection** — auto-mirrors profile shots to always face right
- **Body fat overlay** — integrates BF% data from Apple Health (optional)
- **Simple web viewer** — timeline comparison, front/back/profile views, date navigation
- **Kubernetes-ready** — deploys as a static nginx pod

## How it works

```
Photo (front + back + profile)
        ↓
  ingest.sh
        ↓
  [1] Archive originals
  [2] Update data.json
  [3] align_photos.py
       ├─ MediaPipe pose detection → rotation, scale, crop
       ├─ Force-align pass (segmentation bbox) → pixel-perfect positioning
       └─ rembg background removal → uniform dark background
  [4] Fetch BF% from Apple Health DB (optional)
  [5] Deploy to K8s
```

### Alignment pipeline

1. **Rotation** — levels shoulders (front/back) or corrects body tilt (profile)
2. **Scale** — body height always fills 94% of the frame
3. **Crop** — centered on shoulder midpoint, head at 3% from top
4. **Force-align** — second pass using segmentation bounding box for pixel-perfect consistency
5. **Background removal** — rembg isolates the person, replaces background with `#1E1E1E`
6. **Profile mirror** — auto-detects facing direction, always mirrors to face right

Output: `800×1067px` (3:4), dark grey background.

## Setup

### Requirements

- Python 3.10+
- ~500MB disk for ML models (auto-downloaded on first run)

```bash
git clone https://github.com/jStrider/body-progress-tracker
cd body-progress-tracker

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage

```bash
# Add a new entry (front, back, profile)
./ingest.sh 2026-03-28 /path/to/front.jpg /path/to/back.jpg /path/to/side.jpg

# Re-align all existing photos
source venv/bin/activate
python3 align_photos.py --all

# Re-align a specific date
python3 align_photos.py --date 2026-03-28
```

### Data structure

```
photos/
  2026-03-17/
    front.jpg       # original
    back.jpg
    profile.jpg
    aligned/
      front.jpg     # processed
      back.jpg
      profile.jpg
data.json           # entries with dates, BF%, paths
```

Copy `data.example.json` to `data.json` to get started.

### Apple Health integration (optional)

The ingest script can pull body fat % from a PostgreSQL database populated by Apple Health:

```bash
# In ingest.sh, set KUBECONFIG or adapt the psql call to your setup
```

Expected table: `health_metrics` with columns `metric_name`, `value`, `date`.

### Deploy to Kubernetes

```bash
# The ingest.sh script handles deployment automatically if KUBECONFIG is set
# Manual deploy:
kubectl cp photos/ <pod>:/usr/share/nginx/html/photos/ -n progress-tracker
kubectl cp data.json <pod>:/usr/share/nginx/html/data.json -n progress-tracker
kubectl cp index.html <pod>:/usr/share/nginx/html/index.html -n progress-tracker
```

## Tech stack

- **Python** — alignment pipeline
- **MediaPipe** — pose detection + body segmentation
- **rembg** — background removal (u2net model)
- **OpenCV** — image processing
- **Vanilla JS + HTML** — web viewer (zero dependencies)
- **nginx** — static file serving

## License

MIT

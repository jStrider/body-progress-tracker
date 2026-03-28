"""
Model download helpers for body-align.

Models are cached in ~/.cache/body-align/ and auto-downloaded on first use.
Total download size: ~500MB.
"""

import urllib.request
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "body-align"

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
POSE_MODEL_NAME = "pose_landmarker_full.task"

SEGMENTATION_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
SEGMENTATION_MODEL_NAME = "selfie_multiclass_256x256.tflite"


def _ensure_model(url: str, name: str) -> str:
    """Download model if not cached, return local path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CACHE_DIR / name
    if not model_path.exists():
        print(f"  Downloading {name} (~this may take a moment)...")
        urllib.request.urlretrieve(url, str(model_path))
        print(f"  Saved to {model_path}")
    return str(model_path)


def get_pose_model_path() -> str:
    """Return local path to pose landmarker model, downloading if needed."""
    return _ensure_model(POSE_MODEL_URL, POSE_MODEL_NAME)


def get_segmentation_model_path() -> str:
    """Return local path to segmentation model, downloading if needed."""
    return _ensure_model(SEGMENTATION_MODEL_URL, SEGMENTATION_MODEL_NAME)

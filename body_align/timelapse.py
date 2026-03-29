"""
timelapse.py — generate timelapse MP4 or GIF from aligned photos.
"""
import glob
import re
from pathlib import Path
from typing import Optional, List


def make_timelapse(
    aligned_dir: str,
    views: Optional[List[str]] = None,
    fps: float = 2.0,
    output_path: Optional[str] = None,
    as_gif: bool = False,
) -> str:
    """
    Generate a timelapse video or GIF from aligned photos.

    Args:
        aligned_dir: Directory with YYYY-MM-DD/<view>.jpg structure
        views: Views to include (default: ['front'])
        fps: Frames per second (default: 2)
        output_path: Output path (default: <aligned_dir>/timelapse_<view>.mp4 or .gif)
        as_gif: If True, output GIF instead of MP4

    Returns:
        Path to generated file
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise ImportError("opencv-python required: pip install opencv-python")

    if views is None:
        views = ["front"]

    base = Path(aligned_dir)

    # Collect date dirs in sorted order
    date_dirs = sorted([
        d for d in base.iterdir()
        if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name)
    ])

    if not date_dirs:
        raise ValueError(f"No YYYY-MM-DD directories found in {aligned_dir}")

    # Collect frames
    frames = []
    for date_dir in date_dirs:
        for view in views:
            p = date_dir / f"{view}.jpg"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    frames.append((date_dir.name, view, img))

    if not frames:
        raise ValueError(f"No frames found for views {views} in {aligned_dir}")

    # Ensure consistent size
    target_h, target_w = frames[0][2].shape[:2]
    normalized = []
    for date, view, img in frames:
        h, w = img.shape[:2]
        if h != target_h or w != target_w:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        normalized.append((date, view, img))

    view_str = "-".join(views)
    if output_path is None:
        ext = "gif" if as_gif else "mp4"
        output_path = str(base / f"timelapse_{view_str}.{ext}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if as_gif:
        _write_gif(normalized, output_path, fps, target_w, target_h)
    else:
        _write_mp4(normalized, output_path, fps, target_w, target_h)

    return output_path


def _write_mp4(frames, output_path, fps, w, h):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for date, view, img in frames:
        # Add date label
        labeled = img.copy()
        cv2.putText(
            labeled, f"{date}", (10, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA
        )
        writer.write(labeled)
    writer.release()


def _write_gif(frames, output_path, fps, w, h):
    from PIL import Image as PILImage
    duration = int(1000 / fps)  # ms per frame
    pil_frames = []
    for date, view, img in frames:
        import cv2
        import numpy as np
        labeled = img.copy()
        cv2.putText(
            labeled, f"{date}", (10, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA
        )
        rgb = cv2.cvtColor(labeled, cv2.COLOR_BGR2RGB)
        pil_frames.append(PILImage.fromarray(rgb))

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )

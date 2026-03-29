"""
compare.py — side-by-side comparison of two aligned photos.
"""
from pathlib import Path
from typing import Optional, List

try:
    import cv2
    import numpy as np
except ImportError:
    raise ImportError("opencv-python required: pip install opencv-python")


def make_comparison(
    date1: str,
    date2: str,
    aligned_dir: str,
    views: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a side-by-side comparison image.

    Args:
        date1: First date (YYYY-MM-DD)
        date2: Second date (YYYY-MM-DD)
        aligned_dir: Directory containing YYYY-MM-DD/view.jpg structure
        views: List of views to include (default: front only)
        output_path: Output file path (default: <aligned_dir>/comparison_DATE1_DATE2.jpg)

    Returns:
        Path to the output file
    """
    if views is None:
        views = ["front"]

    base = Path(aligned_dir)
    panels = []

    for date in [date1, date2]:
        for view in views:
            p = base / date / f"{view}.jpg"
            if not p.exists():
                raise FileNotFoundError(f"Photo not found: {p}")
            img = cv2.imread(str(p))
            if img is None:
                raise ValueError(f"Cannot read: {p}")
            panels.append(img)

    # Ensure same height
    target_h = max(p.shape[0] for p in panels)
    resized = []
    for img in panels:
        h, w = img.shape[:2]
        if h != target_h:
            new_w = int(w * target_h / h)
            img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        resized.append(img)

    # Add date label bar on each panel
    label_h = 36
    labeled = []
    for i, img in enumerate(resized):
        h, w = img.shape[:2]
        bar = np.zeros((label_h, w, 3), dtype=np.uint8)
        date_label = date1 if i < len(views) else date2
        view_label = views[i % len(views)]
        text = f"{date_label}  {view_label}"
        cv2.putText(
            bar, text, (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA
        )
        labeled.append(np.vstack([bar, img]))

    comparison = np.hstack(labeled)

    if output_path is None:
        output_path = str(base / f"comparison_{date1}_{date2}.jpg")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return output_path

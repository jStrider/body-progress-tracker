"""
verify.py — check alignment consistency across aligned photos.
"""
import re
from pathlib import Path
from typing import Optional, List


def verify_alignment(
    aligned_dir: str,
    views: Optional[List[str]] = None,
    threshold: float = 20.0,
) -> dict:
    """
    Re-detect poses on all aligned photos and report head/feet variance.

    Args:
        aligned_dir: Directory with YYYY-MM-DD/<view>.jpg structure
        views: Views to check (default: front, back, profile)
        threshold: Warn if position variance > this many pixels (default: 20)

    Returns:
        dict with per-view stats and warnings
    """
    from body_align.align import detect_pose
    from body_align.models import get_pose_model_path

    if views is None:
        views = ["front", "back", "profile"]

    base = Path(aligned_dir)
    date_dirs = sorted([
        d for d in base.iterdir()
        if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name)
    ])

    if not date_dirs:
        return {"error": f"No YYYY-MM-DD directories found in {aligned_dir}"}

    results = {}

    for view in views:
        head_ys = []
        feet_ys = []
        dates = []
        errors = []

        for date_dir in date_dirs:
            p = date_dir / f"{view}.jpg"
            if not p.exists():
                continue

            pose = detect_pose(p)
            if pose is None:
                errors.append(f"{date_dir.name}: no pose detected")
                continue

            head_ys.append(pose["head_top_y"])
            feet_ys.append(pose["feet_bottom_y"])
            dates.append(date_dir.name)

        if not head_ys:
            results[view] = {
                "photos": 0,
                "errors": errors,
                "warnings": [f"No poses detected for {view}"],
            }
            continue

        import statistics
        head_variance = max(head_ys) - min(head_ys) if len(head_ys) > 1 else 0
        feet_variance = max(feet_ys) - min(feet_ys) if len(feet_ys) > 1 else 0
        head_stdev = statistics.stdev(head_ys) if len(head_ys) > 1 else 0
        feet_stdev = statistics.stdev(feet_ys) if len(feet_ys) > 1 else 0

        warnings = []
        if head_variance > threshold:
            warnings.append(
                f"Head Y variance {head_variance:.1f}px > {threshold}px "
                f"(stdev={head_stdev:.1f}px)"
            )
        if feet_variance > threshold:
            warnings.append(
                f"Feet Y variance {feet_variance:.1f}px > {threshold}px "
                f"(stdev={feet_stdev:.1f}px)"
            )

        results[view] = {
            "photos": len(head_ys),
            "dates": dates,
            "head_y": {
                "values": head_ys,
                "min": min(head_ys),
                "max": max(head_ys),
                "range": head_variance,
                "stdev": head_stdev,
            },
            "feet_y": {
                "values": feet_ys,
                "min": min(feet_ys),
                "max": max(feet_ys),
                "range": feet_variance,
                "stdev": feet_stdev,
            },
            "errors": errors,
            "warnings": warnings,
            "ok": len(warnings) == 0,
        }

    return results

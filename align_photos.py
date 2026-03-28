#!/usr/bin/env python3
"""
Align progress photos using MediaPipe pose detection + rembg.

Pipeline:
1. Rotation correction (level shoulders + body tilt via shoulder-hip axis)
2. Full-body normalization: head-to-feet always fills the same vertical range
3. Consistent scale: person's height in pixels is always identical
4. Profile mirror: always facing right
5. Force-align pass: rembg segmentation for pixel-perfect body bbox
6. Background removal: uniform dark background via rembg
7. Output always OUTPUT_W x OUTPUT_H

Usage:
    python3 align_photos.py --all
    python3 align_photos.py --date 2026-03-19
"""

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from PIL import Image as PILImage
    from rembg import remove as rembg_remove
except ImportError:
    print("Missing dependencies: pip install opencv-python mediapipe numpy rembg[cpu] pillow")
    sys.exit(1)

# Background color for final output
BG_COLOR = (30, 30, 30)  # dark grey

# --- Config ---
OUTPUT_W = 800
OUTPUT_H = 1067  # 3:4 ratio

# Margins in the output image (relative to OUTPUT_H)
MARGIN_TOP = 0.03      # 3% above head
MARGIN_BOTTOM = 0.03   # 3% below feet
# So body fills 94% of the frame height

DATA_JSON = Path(__file__).parent / "data.json"
PHOTOS_DIR = Path(__file__).parent / "photos"

# MediaPipe Pose Landmarks
# 0: nose, 7: left_ear, 8: right_ear
# 11: left_shoulder, 12: right_shoulder
# 23: left_hip, 24: right_hip
# 27: left_ankle, 28: right_ankle
# 29: left_heel, 30: right_heel
# 31: left_foot_index, 32: right_foot_index


def _get_pose_model_path() -> str:
    model_path = Path(__file__).parent / "pose_landmarker_full.task"
    if not model_path.exists():
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        print("  Downloading pose model...")
        urllib.request.urlretrieve(url, str(model_path))
    return str(model_path)


def _find_body_edge_top(img, center_x: int, approx_y: int, w: int, h: int) -> float:
    """Scan upward from approx_y to find actual top of head using edge detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Define scan column range (center ± 15% of width)
    col_start = max(0, center_x - int(w * 0.15))
    col_end = min(w, center_x + int(w * 0.15))
    
    # Scan upward from approximate head position
    search_start = max(0, int(approx_y) - 5)  # start slightly above estimate
    
    # Get the background color from the very top rows
    bg_strip = gray[0:min(5, h), col_start:col_end]
    bg_mean = np.mean(bg_strip) if bg_strip.size > 0 else 200
    
    # Scan upward, find first row where the center columns differ significantly from background
    threshold = 35  # pixel intensity difference from background
    for row in range(search_start, -1, -1):
        strip = gray[row, col_start:col_end]
        strip_mean = np.mean(strip)
        if abs(strip_mean - bg_mean) > threshold:
            return float(row)
    
    return float(approx_y)


def _find_body_edge_bottom(img, center_x: int, approx_y: int, w: int, h: int) -> float:
    """Scan downward from approx_y to find actual bottom of feet."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    col_start = max(0, center_x - int(w * 0.2))
    col_end = min(w, center_x + int(w * 0.2))
    
    search_start = min(h - 1, int(approx_y) + 5)
    
    # Get background from bottom rows
    bg_strip = gray[max(0, h - 5):h, col_start:col_end]
    bg_mean = np.mean(bg_strip) if bg_strip.size > 0 else 200
    
    threshold = 35
    for row in range(search_start, h):
        strip = gray[row, col_start:col_end]
        strip_mean = np.mean(strip)
        if abs(strip_mean - bg_mean) < threshold:
            return float(row)
    
    return float(approx_y)


def detect_pose(image_path: Path) -> dict | None:
    """
    Detect full pose. Returns dict with:
    - shoulders: midpoint, angle, distance
    - head_top_y: estimated top of head (pixels)
    - feet_bottom_y: bottom of feet (pixels)
    - nose_x: nose X position (for profile direction)
    - shoulder_mid_x: shoulder midpoint X (for profile direction)
    - all landmark positions
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    base_options = mp_python.BaseOptions(model_asset_path=_get_pose_model_path())
    options = mp_vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        lm = result.pose_landmarks[0]

        # Shoulders
        ls = lm[11]
        rs = lm[12]
        if ls.visibility < 0.15 and rs.visibility < 0.15:
            return None

        lsx, lsy = ls.x * w, ls.y * h
        rsx, rsy = rs.x * w, rs.y * h
        smid_x = (lsx + rsx) / 2
        smid_y = (lsy + rsy) / 2
        sdist = math.sqrt((rsx - lsx) ** 2 + (rsy - lsy) ** 2)

        raw_angle = math.degrees(math.atan2(rsy - lsy, rsx - lsx))
        if raw_angle > 90:
            raw_angle -= 180
        elif raw_angle < -90:
            raw_angle += 180

        is_profile = sdist < 80

        nose = lm[0]
        nose_x, nose_y = nose.x * w, nose.y * h

        # --- HEAD TOP ---
        # Use the topmost visible landmark (nose, eyes, ears) as reference
        # Then add a fixed offset for the skull above that point
        top_landmarks = []
        for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # face landmarks
            pt = lm[idx]
            if pt.visibility > 0.1:
                top_landmarks.append(pt.y * h)
        
        if top_landmarks:
            topmost_face_y = min(top_landmarks)
            # The skull extends ~50% of nose-to-shoulder distance above the topmost landmark
            nose_to_shoulder = smid_y - nose_y
            skull_offset = max(nose_to_shoulder * 0.45, 20)  # at least 20px
            head_top_y = topmost_face_y - skull_offset
        else:
            head_top_y = nose_y - (smid_y - nose_y) * 0.7
        
        head_top_y = max(0, head_top_y)

        # --- FEET BOTTOM ---
        # Use ALL lower body landmarks and take the maximum
        feet_candidates = []
        for idx in [27, 28, 29, 30, 31, 32]:  # ankles, heels, foot index
            pt = lm[idx]
            if pt.visibility > 0.1:
                feet_candidates.append(pt.y * h)
        
        if feet_candidates:
            lowest_point = max(feet_candidates)
            # Add small offset for the sole of the foot below landmarks
            feet_bottom_y = lowest_point + 10
        else:
            # Low visibility fallback: use ankles anyway
            ankle_y = max(lm[27].y * h, lm[28].y * h)
            feet_bottom_y = ankle_y + 30  # rough offset for foot below ankle
        
        feet_bottom_y = min(h, feet_bottom_y)

        # --- BODY TILT (shoulder-hip axis) ---
        lh = lm[23]  # left hip
        rh = lm[24]  # right hip
        lhx, lhy = lh.x * w, lh.y * h
        rhx, rhy = rh.x * w, rh.y * h
        hip_mid_x = (lhx + rhx) / 2
        hip_mid_y = (lhy + rhy) / 2

        # Body tilt = angle of the line from hip midpoint to shoulder midpoint vs vertical
        body_tilt = math.degrees(math.atan2(smid_x - hip_mid_x, hip_mid_y - smid_y))
        # Positive = leaning right, negative = leaning left

        return {
            "shoulder_mid": (smid_x, smid_y),
            "shoulder_angle": raw_angle,
            "shoulder_dist": sdist,
            "is_profile": is_profile,
            "head_top_y": head_top_y,
            "feet_bottom_y": feet_bottom_y,
            "nose_x": nose_x,
            "shoulder_mid_x": smid_x,
            "body_height": feet_bottom_y - head_top_y,
            "body_tilt": body_tilt,
            "hip_mid": (hip_mid_x, hip_mid_y),
            "img_w": w,
            "img_h": h,
        }


def align_and_crop(image_path: Path, output_path: Path, pose: dict | None, view: str):
    """
    Full alignment pipeline:
    1. Rotate to level shoulders
    2. Scale so body height fills the frame consistently
    3. Center horizontally on shoulder midpoint
    4. Pad with black if needed
    5. Mirror profile if facing left
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [SKIP] Cannot read {image_path}")
        return False

    h, w = img.shape[:2]

    if pose is None:
        # Fallback: center crop with black padding
        print(f"  [FALLBACK] No pose for {image_path.name}, center crop")
        result = _center_crop_with_padding(img)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return True

    smid_x, smid_y = pose["shoulder_mid"]
    shoulder_angle = pose["shoulder_angle"]
    body_tilt = pose.get("body_tilt", 0)

    # For profile: shoulders are unreliable (tiny distance), use body tilt instead
    if pose["is_profile"]:
        # Use body tilt (shoulder-hip axis) as primary rotation for profiles
        angle = body_tilt
        print(f"  [{view}] Profile rotation: body_tilt={body_tilt:.1f}°")
    else:
        # Front/back: blend shoulder leveling (60%) + body tilt correction (40%)
        angle = shoulder_angle * 0.6 + body_tilt * 0.4

    # ===== Step 1: Rotate =====
    rot_matrix = cv2.getRotationMatrix2D((smid_x, smid_y), angle, 1.0)

    cos_a = abs(math.cos(math.radians(angle)))
    sin_a = abs(math.sin(math.radians(angle)))
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    rot_matrix[0, 2] += (new_w - w) / 2
    rot_matrix[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(img, rot_matrix, (new_w, new_h),
                              flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))

    # Transform key points through rotation
    def transform_pt(x, y):
        pt = np.array([x, y, 1.0])
        result = rot_matrix @ pt
        return result[0], result[1]

    r_smid_x, r_smid_y = transform_pt(smid_x, smid_y)
    _, r_head_top_y = transform_pt(smid_x, pose["head_top_y"])
    _, r_feet_bottom_y = transform_pt(smid_x, pose["feet_bottom_y"])
    r_nose_x, _ = transform_pt(pose["nose_x"], smid_y)

    body_height_px = r_feet_bottom_y - r_head_top_y
    if body_height_px <= 0:
        body_height_px = new_h * 0.8  # fallback

    # ===== Step 2: Scale =====
    # Body should fill OUTPUT_H * (1 - MARGIN_TOP - MARGIN_BOTTOM)
    target_body_height = OUTPUT_H * (1.0 - MARGIN_TOP - MARGIN_BOTTOM)
    scale = target_body_height / body_height_px

    scaled_w = int(new_w * scale)
    scaled_h = int(new_h * scale)

    scaled = cv2.resize(rotated, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

    # Scale all points
    r_smid_x *= scale
    r_smid_y *= scale
    r_head_top_y *= scale
    r_feet_bottom_y *= scale
    r_nose_x *= scale

    # ===== Step 3: Determine crop region =====
    # Head should be at MARGIN_TOP * OUTPUT_H from top
    # Center horizontally on shoulder midpoint (or body center)
    target_head_y = MARGIN_TOP * OUTPUT_H
    target_center_x = OUTPUT_W / 2

    crop_x0 = int(r_smid_x - target_center_x)
    crop_y0 = int(r_head_top_y - target_head_y)

    # ===== Step 4: Extract with black padding =====
    result = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)

    # Source region (clamped to scaled image bounds)
    src_x0 = max(0, crop_x0)
    src_y0 = max(0, crop_y0)
    src_x1 = min(scaled_w, crop_x0 + OUTPUT_W)
    src_y1 = min(scaled_h, crop_y0 + OUTPUT_H)

    # Destination region in output
    dst_x0 = max(0, -crop_x0)
    dst_y0 = max(0, -crop_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if src_x1 > src_x0 and src_y1 > src_y0:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = scaled[src_y0:src_y1, src_x0:src_x1]

    # ===== Step 5: Mirror profile if facing left =====
    if view == "profile":
        # Determine facing direction: if nose is to the left of shoulder mid → facing left → mirror
        if r_nose_x < r_smid_x:
            result = cv2.flip(result, 1)
            print(f"  [{view}] Mirrored (was facing left)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


def _center_crop_with_padding(img):
    """Fallback: center crop to OUTPUT ratio, pad with black."""
    h, w = img.shape[:2]
    aspect = OUTPUT_W / OUTPUT_H

    if w / h > aspect:
        # Too wide, crop sides
        crop_w = int(h * aspect)
        x0 = (w - crop_w) // 2
        cropped = img[:, x0:x0 + crop_w]
    else:
        # Too tall, crop top/bottom
        crop_h = int(w / aspect)
        y0 = (h - crop_h) // 2
        cropped = img[y0:y0 + crop_h, :]

    return cv2.resize(cropped, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LANCZOS4)


# Store first-pass detection results for force-align
_first_pass_results = {}  # key: (date, view) -> {"head_y": ..., "feet_y": ..., "img_h": ...}


def process_entry(entry: dict) -> bool:
    date = entry.get("date")
    photos = entry.get("photos", {})
    if not photos:
        print(f"[{date}] No photos, skipping.")
        return False

    print(f"[{date}] Processing...")
    aligned_dir = PHOTOS_DIR / date / "aligned"
    any_success = False

    # Always use originals
    orig_photos = {}
    for key in ["front", "back", "profile"]:
        src = photos.get(key)
        if not src:
            continue
        orig_path = src.replace("/aligned/", "/")
        orig_photos[key] = orig_path

    new_photos = {}
    for key, src in orig_photos.items():
        src_path = Path(__file__).parent / src
        if not src_path.exists():
            print(f"  [SKIP] {src} not found")
            new_photos[key] = photos.get(key, src)
            continue

        out_path = aligned_dir / f"{key}.jpg"

        pose = detect_pose(src_path)
        if pose:
            bh = pose['body_height']
            print(f"  [{key}] body_h={bh:.0f}px, shoulders: angle={pose['shoulder_angle']:.1f}°, "
                  f"dist={pose['shoulder_dist']:.0f}px, profile={pose['is_profile']}")
        else:
            print(f"  [{key}] No pose detected, using fallback")

        if align_and_crop(src_path, out_path, pose, key):
            new_photos[key] = f"photos/{date}/aligned/{key}.jpg"
            any_success = True

            # Re-detect on aligned output to get precise positions in output space
            aligned_pose = detect_pose(out_path)
            if aligned_pose:
                _first_pass_results[(date, key)] = {
                    "head_y": aligned_pose["head_top_y"],
                    "feet_y": aligned_pose["feet_bottom_y"],
                    "path": out_path,
                }
        else:
            new_photos[key] = photos.get(key, src)

    if any_success:
        entry["photos"] = new_photos

    return any_success


def _get_segmentation_model_path() -> str:
    model_path = Path(__file__).parent / "selfie_multiclass_256x256.tflite"
    if not model_path.exists():
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
        print("  Downloading segmentation model...")
        urllib.request.urlretrieve(url, str(model_path))
    return str(model_path)


def segment_person_bbox(image_path: Path) -> dict | None:
    """Use MediaPipe segmentation to find exact person bounding box."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    base_options = mp_python.BaseOptions(model_asset_path=_get_segmentation_model_path())
    options = mp_vision.ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=True,
    )

    with mp_vision.ImageSegmenter.create_from_options(options) as segmenter:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = segmenter.segment(mp_image)

        mask = result.category_mask.numpy_view()
        # Categories: 0=background, 1=hair, 2=body_skin, 3=face_skin, 4=clothes, 5=others
        person_mask = (mask.squeeze() > 0).astype(np.uint8)

        rows = np.any(person_mask, axis=1)
        cols = np.any(person_mask, axis=0)

        if not np.any(rows):
            return None

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return {
            "top": int(rmin),
            "bottom": int(rmax),
            "left": int(cmin),
            "right": int(cmax),
            "height": int(rmax - rmin),
            "center_x": int((cmin + cmax) // 2),
        }


def _find_body_top(img, center_x: int, search_from: int) -> int:
    """Scan from top of image downward to find the first row with non-background pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Scan a wide column band around center
    col_start = max(0, center_x - int(w * 0.2))
    col_end = min(w, center_x + int(w * 0.2))
    
    # For each row from top, check if there's significant content
    # Use threshold: if any pixel in the band is bright enough, that's body
    for row in range(0, min(search_from + 50, h)):
        strip = gray[row, col_start:col_end]
        # Check if there are bright pixels (body) vs black (background/padding)
        if np.max(strip) > 30:  # not pure black
            return row
    
    return search_from


def _find_body_bottom(img, center_x: int, search_from: int) -> int:
    """Scan from bottom of image upward to find the last row with non-background pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    col_start = max(0, center_x - int(w * 0.2))
    col_end = min(w, center_x + int(w * 0.2))
    
    for row in range(h - 1, max(search_from - 50, 0), -1):
        strip = gray[row, col_start:col_end]
        if np.max(strip) > 30:
            return row
    
    return search_from


def force_align_view(entries, view: str):
    """
    Second pass: use SEGMENTATION to find exact person bounding box,
    then crop+resize to force head/feet to exact target positions.
    """
    valid = []
    for entry in entries:
        date = entry.get("date")
        aligned_path = PHOTOS_DIR / date / "aligned" / f"{view}.jpg"
        if not aligned_path.exists():
            continue

        bbox = segment_person_bbox(aligned_path)
        if bbox and bbox["height"] > 50:
            valid.append({
                "date": date,
                "path": aligned_path,
                "top": bbox["top"],
                "bottom": bbox["bottom"],
                "center_x": bbox["center_x"],
                "height": bbox["height"],
            })

    if len(valid) < 1:
        return

    target_top = int(MARGIN_TOP * OUTPUT_H)
    target_bottom = int((1 - MARGIN_BOTTOM) * OUTPUT_H)
    target_body_h = target_bottom - target_top

    print(f"\n  [{view}] Force-align (segmentation): target top={target_top}, bottom={target_bottom}, body_h={target_body_h}")

    for det in valid:
        img = cv2.imread(str(det["path"]))
        if img is None:
            continue

        h, w = img.shape[:2]
        body_top = det["top"]
        body_bottom = det["bottom"]
        body_h = body_bottom - body_top

        # Scale the full image so body height matches target
        scale = target_body_h / body_h
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Calculate where body top lands after scaling
        scaled_top = int(body_top * scale)

        # Offset to place body top at target position
        offset_y = target_top - scaled_top
        offset_x = (OUTPUT_W - new_w) // 2

        # Create output
        result = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)

        src_y0 = max(0, -offset_y)
        src_y1 = min(new_h, OUTPUT_H - offset_y)
        dst_y0 = max(0, offset_y)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        src_x0 = max(0, -offset_x)
        src_x1 = min(new_w, OUTPUT_W - offset_x)
        dst_x0 = max(0, offset_x)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        if src_y1 > src_y0 and src_x1 > src_x0:
            result[dst_y0:dst_y1, dst_x0:dst_x1] = scaled[src_y0:src_y1, src_x0:src_x1]

        cv2.imwrite(str(det["path"]), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"    {det['date']}: top {body_top}→{target_top}, bottom {body_bottom}→{target_bottom}, "
              f"body {body_h}→{target_body_h}, scale {scale:.3f}")


def verify_and_report(entries):
    """Verify alignment consistency across all aligned photos."""
    print("\n📏 Verification pass:")
    for view in ['front', 'back', 'profile']:
        poses = []
        for entry in entries:
            date = entry.get("date")
            aligned = PHOTOS_DIR / date / "aligned" / f"{view}.jpg"
            if aligned.exists():
                p = detect_pose(aligned)
                if p:
                    poses.append({
                        "date": date,
                        "head": p["head_top_y"],
                        "feet": p["feet_bottom_y"],
                        "height": p["feet_bottom_y"] - p["head_top_y"],
                    })
        if len(poses) >= 2:
            heads = [p["head"] for p in poses]
            feet = [p["feet"] for p in poses]
            heights = [p["height"] for p in poses]
            print(f"  {view}: head={min(heads):.0f}-{max(heads):.0f}px (Δ{max(heads)-min(heads):.0f}), "
                  f"feet={min(feet):.0f}-{max(feet):.0f}px (Δ{max(feet)-min(feet):.0f}), "
                  f"height={min(heights):.0f}-{max(heights):.0f}px (Δ{max(heights)-min(heights):.0f})")


def remove_backgrounds(entries):
    """Remove background from all aligned photos and replace with uniform dark background."""
    for entry in entries:
        date = entry.get("date")
        for view in ['front', 'back', 'profile']:
            aligned_path = PHOTOS_DIR / date / "aligned" / f"{view}.jpg"
            if not aligned_path.exists():
                continue
            img = PILImage.open(str(aligned_path)).convert('RGBA')
            out = rembg_remove(img)
            bg = PILImage.new('RGBA', out.size, (*BG_COLOR, 255))
            result = PILImage.alpha_composite(bg, out).convert('RGB')
            result.save(str(aligned_path), quality=92)
            print(f"    {date}/{view} ✅")


def main():
    parser = argparse.ArgumentParser(description="Align progress photos")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--date", type=str)
    args = parser.parse_args()

    with open(DATA_JSON, "r") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    if not entries:
        print("No entries in data.json")
        sys.exit(1)

    modified = False
    if args.all:
        for entry in entries:
            if process_entry(entry):
                modified = True
        # Second pass: force-align head/feet across all photos
        if modified:
            print("\n🔧 Force-align pass (crop + resize)...")
            for view in ['front', 'back', 'profile']:
                force_align_view(entries, view)
        # Background removal pass
        print("\n🎨 Background removal (rembg)...")
        remove_backgrounds(entries)
        # Verify
        verify_and_report(entries)
    else:
        found = False
        for entry in entries:
            if entry.get("date") == args.date:
                found = True
                if process_entry(entry):
                    modified = True
                break
        if not found:
            print(f"No entry for {args.date}")
            sys.exit(1)

    if modified:
        with open(DATA_JSON, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("\n✅ data.json updated.")
    else:
        print("\nNo changes.")


if __name__ == "__main__":
    main()

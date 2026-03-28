"""
Core alignment logic for body-align.

Pipeline:
1. Rotation correction (level shoulders + body tilt via shoulder-hip axis)
2. Full-body normalization: head-to-feet always fills the same vertical range
3. Consistent scale: person's height in pixels is always identical
4. Profile mirror: always facing right
5. Force-align pass: segmentation for pixel-perfect body bbox
6. Background removal: uniform background via rembg
7. Output always OUTPUT_W x OUTPUT_H
"""

import math
from pathlib import Path
from datetime import date as Date
from typing import Optional

try:
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from PIL import Image as PILImage
    from rembg import remove as rembg_remove
except ImportError:
    raise ImportError(
        "Missing dependencies. Install with: "
        "pip install opencv-python mediapipe numpy rembg[cpu] pillow"
    )

from body_align.models import get_pose_model_path, get_segmentation_model_path

# --- Defaults ---
DEFAULT_OUTPUT_W = 800
DEFAULT_OUTPUT_H = 1067  # 3:4 ratio
DEFAULT_BG_COLOR = (30, 30, 30)  # dark grey #1E1E1E
MARGIN_TOP = 0.03
MARGIN_BOTTOM = 0.03


def _parse_bg_color(hex_color: str) -> tuple:
    """Parse #RRGGBB hex string to BGR tuple."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)  # BGR for OpenCV


def _parse_size(size_str: str) -> tuple:
    """Parse WxH string to (width, height) tuple."""
    w, h = size_str.lower().split("x")
    return int(w), int(h)


def detect_pose(image_path: Path) -> Optional[dict]:
    """
    Detect full pose. Returns dict with:
    - shoulders: midpoint, angle, distance
    - head_top_y: estimated top of head (pixels)
    - feet_bottom_y: bottom of feet (pixels)
    - nose_x: nose X position (for profile direction)
    - shoulder_mid_x: shoulder midpoint X
    - body_height: feet_bottom_y - head_top_y
    - body_tilt: body lean angle
    - is_profile: True if profile view
    - img_w, img_h: image dimensions
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    base_options = mp_python.BaseOptions(model_asset_path=get_pose_model_path())
    options = mp_vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        lm = result.pose_landmarks[0]

        ls = lm[11]  # left shoulder
        rs = lm[12]  # right shoulder
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

        # Head top estimation
        top_landmarks = []
        for idx in range(0, 11):  # face landmarks
            pt = lm[idx]
            if pt.visibility > 0.1:
                top_landmarks.append(pt.y * h)

        if top_landmarks:
            topmost_face_y = min(top_landmarks)
            nose_to_shoulder = smid_y - nose_y
            skull_offset = max(nose_to_shoulder * 0.45, 20)
            head_top_y = topmost_face_y - skull_offset
        else:
            head_top_y = nose_y - (smid_y - nose_y) * 0.7

        head_top_y = max(0, head_top_y)

        # Feet bottom estimation
        feet_candidates = []
        for idx in [27, 28, 29, 30, 31, 32]:  # ankles, heels, foot index
            pt = lm[idx]
            if pt.visibility > 0.1:
                feet_candidates.append(pt.y * h)

        if feet_candidates:
            lowest_point = max(feet_candidates)
            feet_bottom_y = lowest_point + 10
        else:
            ankle_y = max(lm[27].y * h, lm[28].y * h)
            feet_bottom_y = ankle_y + 30

        feet_bottom_y = min(h, feet_bottom_y)

        # Body tilt (shoulder-hip axis vs vertical)
        lh = lm[23]  # left hip
        rh = lm[24]  # right hip
        lhx, lhy = lh.x * w, lh.y * h
        rhx, rhy = rh.x * w, rh.y * h
        hip_mid_x = (lhx + rhx) / 2
        hip_mid_y = (lhy + rhy) / 2
        body_tilt = math.degrees(math.atan2(smid_x - hip_mid_x, hip_mid_y - smid_y))

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


def _center_crop_with_padding(img, output_w: int, output_h: int):
    """Fallback: center crop to output ratio, pad with black."""
    h, w = img.shape[:2]
    aspect = output_w / output_h

    if w / h > aspect:
        crop_w = int(h * aspect)
        x0 = (w - crop_w) // 2
        cropped = img[:, x0:x0 + crop_w]
    else:
        crop_h = int(w / aspect)
        y0 = (h - crop_h) // 2
        cropped = img[y0:y0 + crop_h, :]

    return cv2.resize(cropped, (output_w, output_h), interpolation=cv2.INTER_LANCZOS4)


def align_and_crop(
    image_path: Path,
    output_path: Path,
    pose: Optional[dict],
    view: str,
    output_w: int = DEFAULT_OUTPUT_W,
    output_h: int = DEFAULT_OUTPUT_H,
) -> bool:
    """
    Full alignment pipeline for a single image.

    Steps:
    1. Rotate to level shoulders / correct body tilt
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
        print(f"  [FALLBACK] No pose for {image_path.name}, center crop")
        result = _center_crop_with_padding(img, output_w, output_h)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return True

    smid_x, smid_y = pose["shoulder_mid"]
    shoulder_angle = pose["shoulder_angle"]
    body_tilt = pose.get("body_tilt", 0)

    if pose["is_profile"]:
        angle = body_tilt
    else:
        angle = shoulder_angle * 0.6 + body_tilt * 0.4

    # Step 1: Rotate
    rot_matrix = cv2.getRotationMatrix2D((smid_x, smid_y), angle, 1.0)
    cos_a = abs(math.cos(math.radians(angle)))
    sin_a = abs(math.sin(math.radians(angle)))
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    rot_matrix[0, 2] += (new_w - w) / 2
    rot_matrix[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(
        img, rot_matrix, (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

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
        body_height_px = new_h * 0.8

    # Step 2: Scale
    target_body_height = output_h * (1.0 - MARGIN_TOP - MARGIN_BOTTOM)
    scale = target_body_height / body_height_px

    scaled_w = int(new_w * scale)
    scaled_h = int(new_h * scale)
    scaled = cv2.resize(rotated, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

    r_smid_x *= scale
    r_smid_y *= scale
    r_head_top_y *= scale
    r_feet_bottom_y *= scale
    r_nose_x *= scale

    # Step 3: Determine crop
    target_head_y = MARGIN_TOP * output_h
    target_center_x = output_w / 2

    crop_x0 = int(r_smid_x - target_center_x)
    crop_y0 = int(r_head_top_y - target_head_y)

    # Step 4: Extract with black padding
    result = np.zeros((output_h, output_w, 3), dtype=np.uint8)

    src_x0 = max(0, crop_x0)
    src_y0 = max(0, crop_y0)
    src_x1 = min(scaled_w, crop_x0 + output_w)
    src_y1 = min(scaled_h, crop_y0 + output_h)

    dst_x0 = max(0, -crop_x0)
    dst_y0 = max(0, -crop_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if src_x1 > src_x0 and src_y1 > src_y0:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = scaled[src_y0:src_y1, src_x0:src_x1]

    # Step 5: Mirror profile if facing left
    if view == "profile" and r_nose_x < r_smid_x:
        result = cv2.flip(result, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


def segment_person_bbox(image_path: Path) -> Optional[dict]:
    """Use MediaPipe segmentation to find exact person bounding box."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    base_options = mp_python.BaseOptions(model_asset_path=get_segmentation_model_path())
    options = mp_vision.ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=True,
    )

    with mp_vision.ImageSegmenter.create_from_options(options) as segmenter:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = segmenter.segment(mp_image)

        mask = result.category_mask.numpy_view()
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


def force_align_single(
    image_path: Path,
    output_w: int = DEFAULT_OUTPUT_W,
    output_h: int = DEFAULT_OUTPUT_H,
) -> bool:
    """
    Second pass: use segmentation to find exact body bbox,
    then crop+resize to force head/feet to exact target positions.
    """
    bbox = segment_person_bbox(image_path)
    if not bbox or bbox["height"] < 50:
        return False

    img = cv2.imread(str(image_path))
    if img is None:
        return False

    h, w = img.shape[:2]
    target_top = int(MARGIN_TOP * output_h)
    target_bottom = int((1 - MARGIN_BOTTOM) * output_h)
    target_body_h = target_bottom - target_top

    body_top = bbox["top"]
    body_bottom = bbox["bottom"]
    body_h = body_bottom - body_top

    scale = target_body_h / body_h
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    scaled_top = int(body_top * scale)
    offset_y = target_top - scaled_top
    offset_x = (output_w - new_w) // 2

    result = np.zeros((output_h, output_w, 3), dtype=np.uint8)

    src_y0 = max(0, -offset_y)
    src_y1 = min(new_h, output_h - offset_y)
    dst_y0 = max(0, offset_y)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    src_x0 = max(0, -offset_x)
    src_x1 = min(new_w, output_w - offset_x)
    dst_x0 = max(0, offset_x)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if src_y1 > src_y0 and src_x1 > src_x0:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = scaled[src_y0:src_y1, src_x0:src_x1]

    cv2.imwrite(str(image_path), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


def remove_background(image_path: Path, bg_color: tuple = DEFAULT_BG_COLOR) -> None:
    """Remove background from image and replace with uniform color."""
    img = PILImage.open(str(image_path)).convert("RGBA")
    out = rembg_remove(img)
    bg = PILImage.new("RGBA", out.size, (*bg_color[::-1], 255))  # BGR->RGB
    result = PILImage.alpha_composite(bg, out).convert("RGB")
    result.save(str(image_path), quality=92)


def align_photos(
    front_path: str,
    back_path: str,
    profile_path: str,
    output_dir: str,
    date: Optional[str] = None,
    options: Optional[dict] = None,
) -> dict:
    """
    Align a set of three progress photos (front, back, profile).

    Args:
        front_path: Path to front photo
        back_path: Path to back photo
        profile_path: Path to profile/side photo
        output_dir: Directory to save aligned photos
        date: Label for output files (default: today's date)
        options: Dict of options:
            - bg_color: hex color string (default: "#1E1E1E")
            - size: "WxH" string (default: "800x1067")
            - no_bg: bool, skip background removal (default: False)

    Returns:
        dict with keys "front", "back", "profile" pointing to output paths,
        and "success" bool.
    """
    from datetime import date as DateClass

    if options is None:
        options = {}

    if date is None:
        date = DateClass.today().isoformat()

    bg_color_hex = options.get("bg_color", "#1E1E1E")
    bg_color = _parse_bg_color(bg_color_hex)

    size_str = options.get("size", f"{DEFAULT_OUTPUT_W}x{DEFAULT_OUTPUT_H}")
    output_w, output_h = _parse_size(size_str)

    no_bg = options.get("no_bg", False)

    out_dir = Path(output_dir) / date
    out_dir.mkdir(parents=True, exist_ok=True)

    views = {
        "front": front_path,
        "back": back_path,
        "profile": profile_path,
    }

    output_paths = {}
    any_success = False

    for view, src in views.items():
        if not src:
            continue
        src_path = Path(src)
        if not src_path.exists():
            print(f"  [SKIP] {src} not found")
            continue

        out_path = out_dir / f"{view}.jpg"

        pose = detect_pose(src_path)
        if pose:
            print(f"  [{view}] body_h={pose['body_height']:.0f}px, "
                  f"angle={pose['shoulder_angle']:.1f}°, tilt={pose['body_tilt']:.1f}°")
        else:
            print(f"  [{view}] No pose detected, using fallback")

        if align_and_crop(src_path, out_path, pose, view, output_w, output_h):
            # Force-align pass
            force_align_single(out_path, output_w, output_h)

            # Background removal
            if not no_bg:
                print(f"  [{view}] Removing background...")
                remove_background(out_path, bg_color)

            output_paths[view] = str(out_path)
            any_success = True

    return {**output_paths, "success": any_success, "date": date}

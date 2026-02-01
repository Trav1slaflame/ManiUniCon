#!/usr/bin/env python3
import cv2 as cv
import numpy as np
from pathlib import Path
import time

def _load_ref_image(ref_dir: Path, index: int):
    img_path = ref_dir / f"{index}.jpg"
    if not img_path.exists():
        print(f"No reference image path: {img_path}")
        return None
    img = cv.imread(str(img_path), cv.IMREAD_COLOR)
    if img is None:
        print(f"Failed to load reference image: {img_path}")
    return img

def run_overlay_viewer(shared_storage, first_frames_dir: str, camera_name: str = "camera_0",
                       ref_index: int = 0, alpha_init: float = 0.7):
    ref_dir = Path(first_frames_dir)
    if not ref_dir.is_dir():
        print(f"First frames directory does not exist: {ref_dir}")
        return

    ref_img = _load_ref_image(ref_dir, ref_index)
    if ref_img is None:
        print("Failed to load initial reference image. Exiting overlay process.")
        return

    win_name = f"Shared Camera Overlay ({camera_name})"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, 1280, 720)

    cv.createTrackbar("alpha", win_name, int(alpha_init * 100), 100, lambda x: None)

    ref_indices = sorted([int(p.stem) for p in ref_dir.glob("*.jpg") if p.stem.isdigit()])
    max_ref_index = ref_indices[-1] if ref_indices else ref_index
    cv.createTrackbar("ref_index", win_name, min(ref_index, max_ref_index), max_ref_index, lambda x: None)

    print("Overlay process started. Interaction instructions:")
    print("  q/ESC exit overlay process")
    print("  trackbars adjust alpha and ref_index")

    try:
        while True:
            data = shared_storage.read_single_camera(camera_name)
            if data is None:
                time.sleep(0.01)
                continue

            rgb = data.color  # RGB
            if rgb is None:
                continue
            bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

            alpha_slider = cv.getTrackbarPos("alpha", win_name)
            alpha = max(0.0, min(1.0, alpha_slider / 100.0))
            beta = 1.0 - alpha

            track_idx = cv.getTrackbarPos("ref_index", win_name)
            if track_idx != ref_index:
                new_ref = _load_ref_image(ref_dir, track_idx)
                if new_ref is not None:
                    ref_img = new_ref
                    ref_index = track_idx
                else:
                    cv.setTrackbarPos("ref_index", win_name, ref_index)

            h, w = bgr.shape[:2]
            ref_resized = cv.resize(ref_img, (w, h), interpolation=cv.INTER_AREA)

            overlay = cv.addWeighted(bgr, alpha, ref_resized, beta, 0.0)
            text = f"ref_index={ref_index} alpha={alpha:.2f}"
            cv.putText(overlay, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv.imshow(win_name, overlay)
            key = cv.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break
    finally:
        cv.destroyWindow(win_name)
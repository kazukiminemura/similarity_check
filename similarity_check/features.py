import os
import os.path as osp
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    YOLO = None  # type: ignore


COCO_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]


def _ensure_dir(path: str) -> None:
    if path and not osp.exists(path):
        os.makedirs(path, exist_ok=True)


def load_model(model_name: str = "yolov8n-pose.pt"):
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is not installed. Please `pip install ultralytics`."
        )
    return YOLO(model_name)


def _select_main_person(kpts_conf: np.ndarray) -> int:
    """Select person index by average keypoint confidence."""
    if kpts_conf.size == 0:
        return -1
    avg_conf = kpts_conf.mean(axis=1)  # (N,)
    return int(avg_conf.argmax())


def _normalize_keypoints(xyn: np.ndarray, conf: np.ndarray) -> Optional[np.ndarray]:
    """
    Normalize 17x2 keypoints to be translation/scale invariant.
    - Use only points with conf > 0 for center/scale.
    - Center: mean of available joints.
    - Scale: max pairwise distance or bbox size among available joints.
    Returns vector (34,) or None if insufficient points.
    """
    if xyn.shape != (17, 2):
        return None
    valid = conf > 0
    if valid.sum() < 4:  # too few points
        return None
    pts = xyn.copy()
    pts[~valid] = np.nan

    # center by mean of valid points
    center = np.nanmean(pts, axis=0)
    pts = pts - center

    # scale by max distance among valid points
    diffs = pts[valid]
    if diffs.shape[0] >= 2:
        dists = np.linalg.norm(
            diffs[None, :, :] - diffs[:, None, :], axis=-1
        )
        scale = np.nanmax(dists)
    else:
        scale = np.nanmax(np.linalg.norm(diffs, axis=-1))
    if not np.isfinite(scale) or scale <= 1e-6:
        return None
    pts = pts / scale

    # fill NaNs with zeros (missing joints)
    pts = np.nan_to_num(pts, nan=0.0)
    return pts.reshape(-1)


def extract_video_features(
    video_path: str,
    model,
    frame_stride: int = 5,
    max_frames: Optional[int] = None,
    progress: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run pose estimation and build a video-level descriptor.
    Returns dict with keys: 'vector' (1D np.ndarray), 'per_frame' (2D), 'count' (int)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    used_features: List[np.ndarray] = []

    pbar = range(total)
    if progress:
        pbar = tqdm(pbar, desc=f"Pose {osp.basename(video_path)}")

    frame_idx = 0
    used = 0
    for _ in pbar:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        # Run model
        results = model(frame, verbose=False)
        if not results:
            continue
        r0 = results[0]
        if r0.keypoints is None or r0.keypoints.xyn is None:
            continue
        xyn = r0.keypoints.xyn  # (N,17,2)
        conf = r0.keypoints.conf  # (N,17)
        if xyn is None or conf is None or len(xyn) == 0:
            continue

        n_idx = _select_main_person(conf)
        if n_idx < 0:
            continue
        feat = _normalize_keypoints(xyn[n_idx], conf[n_idx])
        if feat is None:
            continue
        used_features.append(feat)
        used += 1
        if max_frames is not None and used >= max_frames:
            break

    cap.release()

    if len(used_features) == 0:
        # Return an empty descriptor to signal failure
        return {"vector": np.zeros(102, dtype=np.float32),
                "per_frame": np.zeros((0, 34), dtype=np.float32),
                "count": 0}

    per_frame = np.stack(used_features, axis=0)  # (T, 34)
    mean = per_frame.mean(axis=0)
    std = per_frame.std(axis=0)
    diffs = np.abs(np.diff(per_frame, axis=0))
    motion = diffs.mean(axis=0) if diffs.size else np.zeros_like(mean)
    vector = np.concatenate([mean, std, motion], axis=0).astype(np.float32)
    return {"vector": vector, "per_frame": per_frame.astype(np.float32), "count": per_frame.shape[0]}


def draw_pose_on_frame(frame: np.ndarray, kpts_xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    out = frame.copy()
    # draw keypoints
    for i, (x, y) in enumerate(kpts_xy):
        if i < len(conf) and conf[i] > 0:
            cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)
    # draw skeleton
    for a, b in COCO_SKELETON:
        if a < len(kpts_xy) and b < len(kpts_xy) and conf[a] > 0 and conf[b] > 0:
            pa = (int(kpts_xy[a, 0]), int(kpts_xy[a, 1]))
            pb = (int(kpts_xy[b, 0]), int(kpts_xy[b, 1]))
            cv2.line(out, pa, pb, (255, 128, 0), 2)
    return out


def make_video_thumbnail_with_pose(
    video_path: str, model, frame_index: int = 0
) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = min(max(frame_index, 0), max(total - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    results = model(frame, verbose=False)
    if not results:
        return frame
    r0 = results[0]
    if r0.keypoints is None or r0.keypoints.xy is None or r0.keypoints.conf is None:
        return frame
    xy = r0.keypoints.xy
    conf = r0.keypoints.conf
    idx = _select_main_person(conf)
    if idx < 0:
        return frame
    return draw_pose_on_frame(frame, xy[idx], conf[idx])


def save_feature_cache(cache_dir: str, video_path: str, vector: np.ndarray) -> str:
    _ensure_dir(cache_dir)
    base = osp.splitext(osp.basename(video_path))[0]
    out = osp.join(cache_dir, f"{base}.npz")
    np.savez_compressed(out, vector=vector)
    return out


def load_feature_cache(cache_dir: str, video_path: str) -> Optional[np.ndarray]:
    base = osp.splitext(osp.basename(video_path))[0]
    path = osp.join(cache_dir, f"{base}.npz")
    if not osp.exists(path):
        return None
    try:
        data = np.load(path)
        return data["vector"]
    except Exception:
        return None


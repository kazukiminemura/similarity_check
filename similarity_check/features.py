import os
import os.path as osp
import logging
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO


logger = logging.getLogger("similarity_check.features")


def _openvino_export_if_needed(model_name: str) -> Optional[str]:
    """
    Ensure an OpenVINO model exists for the given Ultralytics model.
    Returns the path to the exported OpenVINO model directory, or None on failure.
    """
    if YOLO is None:
        return None
    # If user already passes an OpenVINO dir or xml path, just return its dir
    if model_name.endswith(".xml") and osp.exists(model_name):
        return osp.dirname(model_name)
    if model_name.endswith("_openvino_model") and osp.isdir(model_name):
        return model_name
    # Otherwise assume a .pt and export target dir next to it
    base, ext = osp.splitext(model_name)
    if ext.lower() == ".pt":
        exported_dir = base + "_openvino_model"
        if osp.isdir(exported_dir):
            return exported_dir
        try:
            logger.info("Exporting Ultralytics model to OpenVINO: %s", model_name)
            YOLO(model_name).export(format="openvino")
            if osp.isdir(exported_dir):
                return exported_dir
        except Exception as ex:  # pragma: no cover
            logger.warning("OpenVINO export failed for %s: %s", model_name, ex)
    # Fallback: if it's a directory already, try as-is
    if osp.isdir(model_name):
        return model_name
    return None


def _map_device_to_ov(device: Optional[str]) -> Optional[str]:
    if not device:
        return None
    d = device.strip().lower()
    if d in ("cuda", "gpu"):
        return "GPU"
    if d in ("cpu",):
        return "CPU"
    if d in ("auto",):
        return "AUTO"
    # passthrough for already-OV device strings
    return device


# No torch backend mapping: OpenVINO only


COCO_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]


def _ensure_dir(path: str) -> None:
    if path and not osp.exists(path):
        os.makedirs(path, exist_ok=True)


def _to_numpy(x):
    """Convert torch.Tensor or similar to numpy array; return None on failure."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch  # type: ignore
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        return np.asarray(x)
    except Exception:
        return None


def load_model(model_name: str = "yolov8n-pose.pt", device: Optional[str] = None):
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is not installed. Please `pip install ultralytics`."
        )
    exported = _openvino_export_if_needed(model_name)
    if not exported:
        # Enforce OpenVINO-only usage
        raise RuntimeError(
            "OpenVINO model not available. Ensure openvino is installed and allow Ultralytics to export the model."
        )
    model = YOLO(exported)
    ov_dev = _map_device_to_ov(device)
    if ov_dev:
        logger.info("Loaded OpenVINO model, device=%s", ov_dev)
    else:
        logger.info("Loaded OpenVINO model with default device")
    return model


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
    device: Optional[str] = None,
    swing_only: bool = False,
    swing_seconds: Optional[float] = 5,
    *,
    speed_threshold: Optional[float] = 0.6,
    min_consecutive_high: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Run pose estimation and build a video-level descriptor.
    Returns dict with keys: 'vector' (1D np.ndarray), 'per_frame' (2D), 'count' (int)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    used_features: List[np.ndarray] = []
    presence_scores: List[float] = []

    # Prepare device param (OpenVINO only)
    ov_dev = _map_device_to_ov(device)

    pbar = range(total)
    if progress:
        pbar = tqdm(pbar, desc=f"Pose {osp.basename(video_path)}")

    frame_idx = 0
    used = 0
    # store wrist keypoints for speed-based swing detection (normalized coords)
    left_wrist_xy: List[Tuple[float, float]] = []
    right_wrist_xy: List[Tuple[float, float]] = []
    left_wrist_conf: List[float] = []
    right_wrist_conf: List[float] = []
    for _ in pbar:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        # Run model (OpenVINO-only)
        try:
            results = model(frame, verbose=False, device=ov_dev)
        except Exception as ex:  # pragma: no cover
            # Some environments mis-route device arg; retry without
            logger.warning("OpenVINO call with device=%s failed: %s; retrying without device", ov_dev, ex)
            results = model(frame, verbose=False)
        if not results:
            continue
        r0 = results[0]
        if r0.keypoints is None or r0.keypoints.xyn is None:
            continue
        xyn = _to_numpy(r0.keypoints.xyn)  # (N,17,2)
        conf = _to_numpy(r0.keypoints.conf)  # (N,17)
        if xyn is None or conf is None or len(xyn) == 0:
            continue

        n_idx = _select_main_person(conf)
        if n_idx < 0:
            continue
        feat = _normalize_keypoints(xyn[n_idx], conf[n_idx])
        if feat is None:
            continue
        used_features.append(feat)
        # collect wrist points (COCO: 9 left_wrist, 10 right_wrist)
        try:
            lw = xyn[n_idx, 9]
            rw = xyn[n_idx, 10]
            lc = float(conf[n_idx, 9])
            rc = float(conf[n_idx, 10])
        except Exception:
            lw = np.array([np.nan, np.nan], dtype=np.float32)
            rw = np.array([np.nan, np.nan], dtype=np.float32)
            lc = 0.0
            rc = 0.0
        left_wrist_xy.append((float(lw[0]), float(lw[1])))
        right_wrist_xy.append((float(rw[0]), float(rw[1])))
        left_wrist_conf.append(lc if np.isfinite(lc) else 0.0)
        right_wrist_conf.append(rc if np.isfinite(rc) else 0.0)
        # presence score: ratio of valid joints times their mean confidence
        try:
            cj = conf[n_idx]
            valid_j = cj > 0
            coverage = float(valid_j.mean()) if valid_j.size else 0.0
            mean_c = float(cj[valid_j].mean()) if valid_j.any() else 0.0
            presence_scores.append(max(0.0, min(1.0, coverage * mean_c)))
        except Exception:
            presence_scores.append(0.0)
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

    # Optionally select a swing-only window.
    window_start_sec: Optional[float] = None
    window_end_sec: Optional[float] = None
    if swing_only and per_frame.shape[0] > 4:
        # Compute wrist speed per-sample (normalized units per second)
        samples_per_second = max(1.0, fps / max(1, frame_stride))
        lw = np.asarray(left_wrist_xy, dtype=np.float32)
        rw = np.asarray(right_wrist_xy, dtype=np.float32)
        lc = np.asarray(left_wrist_conf, dtype=np.float32)
        rc = np.asarray(right_wrist_conf, dtype=np.float32)

        # distances between consecutive samples; mask with confidences
        def pairwise_speed(xy, c):
            if xy.shape[0] < 2:
                return np.zeros((0,), dtype=np.float32)
            d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            valid = np.minimum(c[:-1] > 0, c[1:] > 0)
            d[~valid] = 0.0
            return d * samples_per_second

        sp_l = pairwise_speed(lw, lc)
        sp_r = pairwise_speed(rw, rc)
        speed = np.maximum(sp_l, sp_r)  # (T-1,)

        # Smooth with ~0.30s moving average to reduce noise
        k = max(3, int(round(samples_per_second * 0.30)))
        if k % 2 == 0:
            k += 1
        if speed.shape[0] >= k:
            kernel = np.ones(k, dtype=np.float32) / float(k)
            speed_s = np.convolve(speed, kernel, mode='same')
        else:
            speed_s = speed

        # Determine swing peak; fall back to presence when wrist speed is unreliable
        presence_arr = np.asarray(presence_scores, dtype=np.float32)
        if presence_arr.size:
            presence_arr = presence_arr[: per_frame.shape[0]]
        peak_idx = 0
        max_speed = 0.0
        positive_samples = 0
        if speed_s.size:
            peak_idx = int(np.argmax(speed_s))
            max_speed = float(speed_s[peak_idx])
            positive_samples = int(np.count_nonzero(speed_s > 1e-3))
        reliable_peak = speed_s.size > 0 and max_speed > 1e-3

        if reliable_peak and speed_threshold is not None:
            thr = float(speed_threshold)
            high_mask = speed_s > thr
            if high_mask.any():
                if min_consecutive_high > 1:
                    best_len = 0
                    best_center = peak_idx
                    run_len = 0
                    for i, flag in enumerate(high_mask):
                        if flag:
                            run_len += 1
                            if run_len > best_len:
                                best_len = run_len
                                best_center = i - run_len // 2
                        else:
                            run_len = 0
                    if best_len >= min_consecutive_high:
                        peak_idx = int(best_center)
                    elif positive_samples < min_consecutive_high:
                        reliable_peak = False
            else:
                reliable_peak = False

        if not reliable_peak:
            if presence_arr.size:
                peak_idx = int(np.argmax(presence_arr))
            else:
                peak_idx = per_frame.shape[0] // 2

        peak_idx = max(0, min(peak_idx, per_frame.shape[0] - 1))

        # Compute window length in samples
        if swing_seconds is None:
            swing_seconds = 5
        win = int(max(5, round(samples_per_second * swing_seconds)))

        # Put ~40% of the window before the peak and 60% after
        pre = int(round(0.4 * win))
        start = max(0, peak_idx - pre)
        end = min(per_frame.shape[0], start + win)

        # Convert indices to seconds
        window_start_sec = float(start * frame_stride) / float(max(1.0, fps))
        window_end_sec = float(end * frame_stride) / float(max(1.0, fps))
        per_frame = per_frame[start:end]
    mean = per_frame.mean(axis=0)
    std = per_frame.std(axis=0)
    diffs = np.abs(np.diff(per_frame, axis=0))
    motion = diffs.mean(axis=0) if diffs.size else np.zeros_like(mean)
    vector = np.concatenate([mean, std, motion], axis=0).astype(np.float32)
    info: Dict[str, np.ndarray] = {
        "vector": vector,
        "per_frame": per_frame.astype(np.float32),
        "count": per_frame.shape[0],
    }
    # attach window metadata (seconds) if available
    if window_start_sec is not None and window_end_sec is not None:
        info["window_start_sec"] = np.asarray(window_start_sec, dtype=np.float32)  # type: ignore
        info["window_end_sec"] = np.asarray(window_end_sec, dtype=np.float32)  # type: ignore
        info["fps"] = np.asarray(fps, dtype=np.float32)  # type: ignore
        info["frame_stride"] = np.asarray(frame_stride, dtype=np.float32)  # type: ignore
    return info


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
    video_path: str, model, frame_index: int = 0, device: Optional[str] = None
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
    ov_dev = _map_device_to_ov(device)
    try:
        results = model(frame, verbose=False, device=ov_dev)
    except Exception as ex:  # pragma: no cover
        logger.warning("OpenVINO thumbnail call with device=%s failed: %s; retrying without device", ov_dev, ex)
        results = model(frame, verbose=False)
    if not results:
        return frame
    r0 = results[0]
    if r0.keypoints is None or r0.keypoints.xy is None or r0.keypoints.conf is None:
        return frame
    xy = _to_numpy(r0.keypoints.xy)
    conf = _to_numpy(r0.keypoints.conf)
    if xy is None or conf is None or len(xy) == 0:
        return frame
    idx = _select_main_person(conf)
    if idx < 0:
        return frame
    return draw_pose_on_frame(frame, xy[idx], conf[idx])


def save_feature_cache(
    cache_dir: str,
    video_path: str,
    vector: np.ndarray,
    *,
    window_start_sec: Optional[float] = None,
    window_end_sec: Optional[float] = None,
    frame_stride: Optional[int] = None,
    fps: Optional[float] = None,
) -> str:
    _ensure_dir(cache_dir)
    base = osp.splitext(osp.basename(video_path))[0]
    out = osp.join(cache_dir, f"{base}.npz")
    np.savez_compressed(
        out,
        vector=vector,
        window_start_sec=(window_start_sec if window_start_sec is not None else -1.0),
        window_end_sec=(window_end_sec if window_end_sec is not None else -1.0),
        frame_stride=(frame_stride if frame_stride is not None else -1),
        fps=(fps if fps is not None else -1.0),
    )
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


def load_feature_meta(cache_dir: str, video_path: str) -> Dict[str, Optional[float]]:
    base = osp.splitext(osp.basename(video_path))[0]
    path = osp.join(cache_dir, f"{base}.npz")
    meta: Dict[str, Optional[float]] = {
        "window_start_sec": None,
        "window_end_sec": None,
        "frame_stride": None,  # type: ignore
        "fps": None,
    }
    if not osp.exists(path):
        return meta
    try:
        data = np.load(path)
        ws = float(data.get("window_start_sec", -1.0))
        we = float(data.get("window_end_sec", -1.0))
        fs = int(data.get("frame_stride", -1))
        fps = float(data.get("fps", -1.0))
        if ws >= 0 and we >= 0 and we >= ws:
            meta["window_start_sec"] = ws
            meta["window_end_sec"] = we
        if fs >= 0:
            meta["frame_stride"] = float(fs)
        if fps >= 0:
            meta["fps"] = fps
        return meta
    except Exception:
        return meta

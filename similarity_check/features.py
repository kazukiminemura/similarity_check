import os
import os.path as osp
import logging
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    YOLO = None  # type: ignore


logger = logging.getLogger("similarity_check.features")
BACKEND_OPENVINO: bool = False


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


def _map_device_to_torch(device: Optional[str]) -> Optional[str]:
    """Map frontend device to torch/ultralytics expected strings.
    Returns 'cpu' or CUDA index string like '0'.
    """
    try:
        import torch  # type: ignore
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False

    if not device:
        return '0' if has_cuda else 'cpu'
    d = device.strip().lower()
    if d in ("gpu", "cuda"):
        return '0' if has_cuda else 'cpu'
    if d in ("auto",):
        return '0' if has_cuda else 'cpu'
    if d in ("cpu",):
        return 'cpu'
    # passthrough for numeric devices
    return d


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
    global BACKEND_OPENVINO
    exported = _openvino_export_if_needed(model_name)
    if exported:
        BACKEND_OPENVINO = True
        model = YOLO(exported)
        ov_dev = _map_device_to_ov(device)
        if ov_dev:
            logger.info("Loaded OpenVINO model, device=%s", ov_dev)
        else:
            logger.info("Loaded OpenVINO model with default device")
        return model
    # Fallback to PyTorch backend safely
    BACKEND_OPENVINO = False
    model = YOLO(model_name)
    torch_dev = _map_device_to_torch(device)
    try:
        model.to(torch_dev)
        logger.warning("OpenVINO unavailable; using torch backend on %s", torch_dev)
    except Exception as ex:
        logger.warning("Failed to move torch model to %s: %s (continuing)", torch_dev, ex)
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
    swing_seconds: Optional[float] = 2.5,
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

    # Prepare device param for backend
    ov_dev = _map_device_to_ov(device) if BACKEND_OPENVINO else None
    torch_dev = _map_device_to_torch(device) if not BACKEND_OPENVINO else None

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
        if BACKEND_OPENVINO:
            results = model(frame, verbose=False, device=ov_dev)
        else:
            # For torch backend, device is controlled by model.to(); just call
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

    # Optionally select a swing-only window by motion peak
    if swing_only and per_frame.shape[0] > 4:
        diffs = np.abs(np.diff(per_frame, axis=0))  # (T-1, 34)
        energy = diffs.mean(axis=1)  # (T-1,)
        peak = int(np.argmax(energy))  # center between frames peak and peak+1
        if swing_seconds is None:
            swing_seconds = 2.5
        # samples_per_second under the given stride
        samples_per_second = max(1.0, fps / max(1, frame_stride))
        win = int(max(5, min(per_frame.shape[0], round(samples_per_second * swing_seconds))))
        half = max(2, win // 2)
        start = max(0, peak - half)
        end = min(per_frame.shape[0], start + win)
        start = max(0, end - win)  # ensure exact window length
        logger.debug(
            "Swing window: fps=%.2f stride=%d samples/s=%.2f win=%d idx=[%d:%d]",
            fps, frame_stride, samples_per_second, end - start, start, end,
        )
        per_frame = per_frame[start:end]
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
    if BACKEND_OPENVINO:
        ov_dev = _map_device_to_ov(device)
        results = model(frame, verbose=False, device=ov_dev)
    else:
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

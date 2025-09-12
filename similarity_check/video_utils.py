import os.path as osp
from typing import List, Tuple, Optional

import cv2
import numpy as np


def _safe_fps(cap: cv2.VideoCapture, default: float = 25.0) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not np.isfinite(fps) or fps <= 1e-3:
        return default
    return float(fps)


def _frame_count(cap: cv2.VideoCapture) -> int:
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return n if n > 0 else 0


def _duration_seconds(cap: cv2.VideoCapture) -> float:
    fps = _safe_fps(cap)
    n = _frame_count(cap)
    return n / fps if fps > 0 else 0.0


def _fit_into_canvas(img: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale = min(dst_w / w, dst_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    x0 = (dst_w - new_w) // 2
    y0 = (dst_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def make_sync_grid_video(
    target_path: str,
    candidate_paths: List[str],
    out_path: str = "results_sync_grid.mp4",
    rows: int = 2,
    cols: int = 3,
    cell_size: Optional[Tuple[int, int]] = None,
    fps: Optional[float] = None,
    max_seconds: Optional[float] = None,
) -> str:
    """
    Create a synchronized grid video: target in top-left, followed by up to 5 candidates.
    Returns the output video path.
    """
    paths = [target_path] + list(candidate_paths[: rows * cols - 1])
    # Open captures
    caps = []
    for p in paths:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            # Fill with dummy capture by using a single black frame
            caps.append(None)
        else:
            caps.append(cap)

    # Determine fps and duration
    src_fps = [(_safe_fps(c) if c is not None else 25.0) for c in caps]
    out_fps = float(min(src_fps)) if fps is None else float(fps)
    src_durs = [(_duration_seconds(c) if c is not None else 0.0) for c in caps]
    duration = float(min(src_durs))
    if max_seconds is not None:
        duration = min(duration, float(max_seconds))
    if duration <= 0:
        duration = 1.0

    # Determine cell size
    if cell_size is None:
        # Use first readable frame size or fallback 320x180
        cell_w, cell_h = 320, 180
        for c in caps:
            if c is None:
                continue
            ret, fr = c.read()
            if ret and fr is not None:
                h, w = fr.shape[:2]
                if h > 0 and w > 0:
                    # Keep 16:9 aspect if possible
                    cell_w = 320
                    cell_h = int(cell_w * h / max(1, w))
                break
        # reset to frame 0 for those we advanced
        for c in caps:
            if c is not None:
                c.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        cell_w, cell_h = cell_size

    grid_w = cols * cell_w
    grid_h = rows * cell_h

    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (grid_w, grid_h))
    total_frames = int(out_fps * duration)

    # Precompute index mapping per out frame
    src_indices = []
    for i, c in enumerate(caps):
        fps_i = src_fps[i]
        n_i = _frame_count(c) if c is not None else 1
        idxs = [min(n_i - 1, max(0, int(round(t * fps_i / out_fps)))) for t in range(total_frames)]
        src_indices.append(idxs)

    # Render frames
    last_frames: List[np.ndarray] = [None] * len(caps)  # type: ignore
    for t in range(total_frames):
        tiles = []
        for i, c in enumerate(caps):
            frame = None
            idx = src_indices[i][t]
            if c is not None:
                c.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, fr = c.read()
                if ok and fr is not None:
                    frame = fr
            if frame is None:
                frame = last_frames[i]
            if frame is None:
                frame = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            # fit to cell
            tile = _fit_into_canvas(frame, cell_w, cell_h)
            last_frames[i] = tile
            tiles.append(tile)

        # compose grid
        rows_imgs = []
        k = 0
        for r in range(rows):
            row = np.hstack(tiles[k:k + cols])
            rows_imgs.append(row)
            k += cols
        grid = np.vstack(rows_imgs)

        # overlay labels
        for i, p in enumerate(paths):
            r = i // cols
            c = i % cols
            x0 = c * cell_w
            y0 = r * cell_h
            label = "Target" if i == 0 else f"Top {i}"
            name = osp.basename(p)
            cv2.rectangle(grid, (x0, y0), (x0 + cell_w, y0 + 28), (0, 0, 0), -1)
            cv2.putText(grid, f"{label}: {name}", (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(grid)

    writer.release()
    for c in caps:
        if c is not None:
            c.release()
    return out_path
import os
import os.path as osp
import shutil
import subprocess
from typing import Optional

import cv2


def ensure_dir(path: str) -> None:
    if path and not osp.isdir(path):
        os.makedirs(path, exist_ok=True)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_clip_ffmpeg(src_path: str, start_sec: float, end_sec: float, out_path: str) -> bool:
    try:
        # Fast path: stream copy with moov relocation for web playback
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{max(0.0, start_sec):.3f}",
            "-to", f"{max(0.0, end_sec):.3f}",
            "-i", src_path,
            "-c", "copy",
            "-movflags", "+faststart",
            out_path,
        ]
        completed = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return completed.returncode == 0 and osp.exists(out_path) and osp.getsize(out_path) > 0
    except Exception:
        return False


def make_video_clip(src_path: str, start_sec: float, end_sec: float, out_dir: str, basename: Optional[str] = None) -> Optional[str]:
    """
    Save a clipped segment [start_sec, end_sec] from src_path into out_dir.
    Returns the output file path on success, otherwise None.
    """
    try:
        if start_sec is None or end_sec is None:
            return None
        if end_sec <= start_sec:
            return None
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            return None

        start_f = max(0, int(round(start_sec * fps)))
        end_f = max(start_f + 1, int(round(end_sec * fps)))

        # Prepare writer
        ensure_dir(out_dir)
        base = basename or osp.splitext(osp.basename(src_path))[0]
        name = f"{base}_clip_{start_f}_{end_f}.mp4"
        out_path = osp.join(out_dir, name)

        # If clip already exists, return it
        if osp.exists(out_path) and osp.getsize(out_path) > 0:
            cap.release()
            return out_path

        # Prefer ffmpeg if present for H.264 copy without re-encode
        ensure_dir(out_dir)
        base = basename or osp.splitext(osp.basename(src_path))[0]
        name = f"{base}_clip_{start_f}_{end_f}.mp4"
        out_path = osp.join(out_dir, name)

        if _ffmpeg_available():
            if _make_clip_ffmpeg(src_path, start_sec, end_sec, out_path):
                cap.release()
                return out_path

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # fallback
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        fidx = start_f
        ok_any = False
        while fidx < end_f:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            ok_any = True
            fidx += 1

        writer.release()
        cap.release()
        if not ok_any:
            try:
                os.remove(out_path)
            except Exception:
                pass
            return None
        return out_path
    except Exception:
        return None

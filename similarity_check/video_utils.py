import logging
import os
import os.path as osp
import shutil
import subprocess
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("similarity_check.video_utils")


def _format_time(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def ensure_dir(path: str) -> None:
    if path and not osp.isdir(path):
        os.makedirs(path, exist_ok=True)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_clip_ffmpeg(src_path: str, start_sec: float, end_sec: float, out_path: str) -> bool:
    try:
        duration = max(0.0, end_sec - start_sec)
        if duration <= 0:
            logger.debug(
                "ffmpeg clip request has non-positive duration for %s (start=%s end=%s)",
                src_path,
                _format_time(start_sec),
                _format_time(end_sec),
            )
            return False
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(0.0, start_sec):.3f}",
            "-i",
            src_path,
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            "-loglevel",
            "error",
            out_path,
        ]
        logger.debug(
            "Running ffmpeg clip command for %s (window %s-%s): %s",
            src_path,
            _format_time(start_sec),
            _format_time(end_sec),
            " ".join(cmd),
        )
        completed = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        success = completed.returncode == 0 and osp.exists(out_path) and osp.getsize(out_path) > 0
        if success:
            logger.debug("ffmpeg clip command succeeded for %s", out_path)
        else:
            logger.debug(
                "ffmpeg clip command failed for %s (returncode=%s)",
                out_path,
                completed.returncode,
            )
        return success
    except Exception:
        logger.exception("ffmpeg clip command raised an error for %s", src_path)
        return False


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
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
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
    """Create a synchronized grid video and return the output path."""
    paths = [target_path] + list(candidate_paths[: rows * cols - 1])

    caps: List[Optional[cv2.VideoCapture]] = []
    for p in paths:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            caps.append(None)
        else:
            caps.append(cap)

    src_fps = [_safe_fps(c) if c is not None else 25.0 for c in caps]
    out_fps = float(min(src_fps)) if fps is None else float(fps)
    src_durs = [_duration_seconds(c) if c is not None else 0.0 for c in caps]
    duration = float(min(src_durs))
    if max_seconds is not None:
        duration = min(duration, float(max_seconds))
    if duration <= 0:
        duration = 1.0

    if cell_size is None:
        cell_w, cell_h = 320, 180
        for c in caps:
            if c is None:
                continue
            ret, fr = c.read()
            if ret and fr is not None:
                h, w = fr.shape[:2]
                if h > 0 and w > 0:
                    cell_w = 320
                    cell_h = int(cell_w * h / max(1, w))
                break
        for c in caps:
            if c is not None:
                c.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        cell_w, cell_h = cell_size

    grid_w = cols * cell_w
    grid_h = rows * cell_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (grid_w, grid_h))
    total_frames = int(out_fps * duration)

    src_indices: List[List[int]] = []
    for i, c in enumerate(caps):
        fps_i = src_fps[i]
        n_i = _frame_count(c) if c is not None else 1
        idxs = [min(n_i - 1, max(0, int(round(t * fps_i / out_fps)))) for t in range(total_frames)]
        src_indices.append(idxs)

    last_frames: List[Optional[np.ndarray]] = [None] * len(caps)
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
            tile = _fit_into_canvas(frame, cell_w, cell_h)
            last_frames[i] = tile
            tiles.append(tile)

        rows_imgs = []
        k = 0
        for r in range(rows):
            row = np.hstack(tiles[k : k + cols])
            rows_imgs.append(row)
            k += cols
        grid = np.vstack(rows_imgs)

        for i, p in enumerate(paths):
            r = i // cols
            c = i % cols
            x0 = c * cell_w
            y0 = r * cell_h
            label = "Target" if i == 0 else f"Top {i}"
            name = osp.basename(p)
            cv2.rectangle(grid, (x0, y0), (x0 + cell_w, y0 + 28), (0, 0, 0), -1)
            cv2.putText(
                grid,
                f"{label}: {name}",
                (x0 + 8, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        writer.write(grid)

    writer.release()
    for c in caps:
        if c is not None:
            c.release()
    return out_path


def make_video_clip(
    src_path: str,
    start_sec: float,
    end_sec: float,
    out_dir: str,
    basename: Optional[str] = None,
) -> Optional[str]:
    """Save a clipped segment [start_sec, end_sec] from src_path into out_dir."""
    cap: Optional[cv2.VideoCapture] = None
    writer = None
    out_path: Optional[str] = None
    try:
        logger.debug(
            "make_video_clip called src_path=%s start_sec=%s end_sec=%s out_dir=%s basename=%s",
            src_path,
            start_sec,
            end_sec,
            out_dir,
            basename,
        )
        start_fmt = _format_time(start_sec)
        end_fmt = _format_time(end_sec)
        window_label = f"{start_fmt}-{end_fmt}"

        if start_sec is None or end_sec is None:
            logger.info(
                "Skipping clip creation for %s: missing window (%s)",
                src_path,
                window_label,
            )
            return None
        if end_sec <= start_sec:
            logger.info(
                "Skipping clip creation for %s: invalid window (%s)",
                src_path,
                window_label,
            )
            return None

        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            logger.warning("Failed to open source video for clip creation: %s", src_path)
            cap.release()
            cap = None
            return None

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            logger.warning(
                "Invalid video dimensions for clip creation: %s (width=%s height=%s)",
                src_path,
                width,
                height,
            )
            cap.release()
            cap = None
            return None

        start_f = max(0, int(round(start_sec * fps)))
        end_f = max(start_f + 1, int(round(end_sec * fps)))

        ensure_dir(out_dir)
        base = basename or osp.splitext(osp.basename(src_path))[0]
        name = f"{base}_clip.mp4"
        out_path = osp.join(out_dir, name)

        if osp.exists(out_path):
            try:
                os.remove(out_path)
                logger.debug("Removed pre-existing clip file: %s", out_path)
            except Exception as err:
                logger.debug("Could not remove pre-existing clip file %s: %s", out_path, err)

        if _ffmpeg_available():
            logger.debug(
                "Attempting to create clip with ffmpeg for %s (window %s)",
                src_path,
                window_label,
            )
            if _make_clip_ffmpeg(src_path, start_sec, end_sec, out_path):
                cap.release()
                cap = None
                logger.info(
                    "Created clip via ffmpeg: %s (window %s)",
                    out_path,
                    window_label,
                )
                return out_path
            logger.debug(
                "ffmpeg clip creation failed for %s; falling back to OpenCV",
                src_path,
            )
        else:
            logger.debug("ffmpeg not available; using OpenCV writer for %s", src_path)

        writer = None
        writer_fourcc = None
        for fourcc_name in ("avc1", "H264", "h264", "X264"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            if writer.isOpened():
                writer_fourcc = fourcc_name
                break
            writer.release()
            writer = None
        if writer is None:
            logger.warning(
                "OpenCV H.264 encoder unavailable for %s; skipping clip generation",
                src_path,
            )
            cap.release()
            cap = None
            return None
        logger.debug("OpenCV VideoWriter initialised with %s for %s", writer_fourcc, out_path)

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
        writer = None
        cap.release()
        cap = None

        if not ok_any:
            try:
                if out_path and osp.exists(out_path):
                    os.remove(out_path)
            except Exception as err:
                logger.debug("Could not remove empty clip file %s: %s", out_path, err)
            logger.warning(
                "Failed to create clip via OpenCV for %s (window %s): no frames written",
                src_path,
                window_label,
            )
            return None

        logger.info(
            "Created clip via OpenCV: %s (window %s)",
            out_path,
            window_label,
        )
        return out_path
    except Exception:
        logger.exception("Unexpected error while creating clip for %s", src_path)
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if out_path and osp.exists(out_path):
            try:
                os.remove(out_path)
            except Exception:
                pass
        return None



"""FastAPI application exposing similarity search endpoints."""

import os
import os.path as osp
import logging
import functools
import inspect
import glob
from typing import List, Tuple, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from similarity_check.features import (
    load_model,
    extract_video_features,
    load_feature_cache,
    save_feature_cache,
    load_feature_meta,
)
from similarity_check.similarity import rank_similar
from similarity_check.video_utils import make_video_clip


# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("similarity_check.web_api")
if not logger.handlers:
    # Basic configuration if the app doesn't configure logging itself
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))


def _shorten(value, maxlen: int = 200):
    """Convert arbitrary values to a truncated string for logging."""
    try:
        s = str(value)
    except Exception:
        return "<unprintable>"
    if len(s) > maxlen:
        return s[: maxlen - 3] + "..."
    return s


def debug_log(func):
    """Decorator to log function entry/exit and exceptions (sync/async)."""
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def _awrap(*args, **kwargs):
            logger.debug("ENTER %s args=%s kwargs=%s", func.__name__, _shorten(args), _shorten(kwargs))
            try:
                result = await func(*args, **kwargs)
                logger.debug("EXIT  %s -> %s", func.__name__, _shorten(result))
                return result
            except Exception:
                logger.exception("ERROR in %s", func.__name__)
                raise
        return _awrap
    else:
        @functools.wraps(func)
        def _wrap(*args, **kwargs):
            logger.debug("ENTER %s args=%s kwargs=%s", func.__name__, _shorten(args), _shorten(kwargs))
            try:
                result = func(*args, **kwargs)
                logger.debug("EXIT  %s -> %s", func.__name__, _shorten(result))
                return result
            except Exception:
                logger.exception("ERROR in %s", func.__name__)
                raise
        return _wrap


BASE_DIR = osp.abspath(osp.join(osp.dirname(__file__), os.pardir))

@debug_log
def _resolve_root(env_name: str, default_rel: str) -> str:
    """Resolve a root directory from configuration and defaults."""
    # If env is set and absolute, use as-is. If relative or not set, resolve from BASE_DIR.
    val = os.environ.get(env_name)
    if val:
        return val if osp.isabs(val) else osp.abspath(osp.join(BASE_DIR, val))
    return osp.abspath(osp.join(BASE_DIR, default_rel))

TARGET_ROOT = _resolve_root("TARGET_ROOT", "target")
REFERENCE_ROOT = _resolve_root("REFERENCE_ROOT", "reference")
STATIC_DIR = osp.join(osp.dirname(__file__), "static")
TEMPLATE_DIR = osp.join(osp.dirname(__file__), "templates")
CLIP_DIR = _resolve_root("CLIP_DIR", "similarity_check_clips")

app = FastAPI(title="Pose Similarity (YOLOv8-Pose) API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if osp.isdir(TARGET_ROOT):
    app.mount("/videos/target", StaticFiles(directory=TARGET_ROOT), name="videos_target")
if osp.isdir(REFERENCE_ROOT):
    app.mount("/videos/reference", StaticFiles(directory=REFERENCE_ROOT), name="videos_reference")
if not osp.isdir(CLIP_DIR):
    os.makedirs(CLIP_DIR, exist_ok=True)
app.mount("/clips", StaticFiles(directory=CLIP_DIR), name="clips")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

_model = None
logger.info("Configured TARGET_ROOT=%s", TARGET_ROOT)
logger.info("Configured REFERENCE_ROOT=%s", REFERENCE_ROOT)
logger.info("Configured CLIP_DIR=%s", CLIP_DIR)


@debug_log
def _ensure_model():
    """Load the pose estimation model once and memoise it globally."""
    global _model
    if _model is None:
        _model = load_model("yolov8n-pose.pt")
    return _model


@debug_log
def _is_video(fname: str) -> bool:
    """Return True when the file name looks like a supported video."""
    f = fname.lower()
    return f.endswith((".mp4", ".mov", ".avi", ".mkv", ".m4v"))


@debug_log
def _list_videos(root: str) -> List[str]:
    """Enumerate available video files below the provided directory."""
    if not osp.isdir(root):
        return []
    return sorted([f for f in os.listdir(root) if _is_video(f)])


@debug_log
def _rel_url(path: str) -> str:
    """Convert a filesystem path into the URL served by FastAPI mounts."""
    # Convert absolute/relative file path to a mounted static URL
    ap = osp.abspath(path)
    if osp.isdir(TARGET_ROOT) and osp.commonpath([ap, TARGET_ROOT]) == TARGET_ROOT:
        rel = osp.relpath(ap, TARGET_ROOT).replace("\\", "/")
        return f"/videos/target/{rel}"
    if osp.isdir(REFERENCE_ROOT) and osp.commonpath([ap, REFERENCE_ROOT]) == REFERENCE_ROOT:
        rel = osp.relpath(ap, REFERENCE_ROOT).replace("\\", "/")
        return f"/videos/reference/{rel}"
    if osp.isdir(CLIP_DIR) and osp.commonpath([ap, CLIP_DIR]) == CLIP_DIR:
        rel = osp.relpath(ap, CLIP_DIR).replace("\\", "/")
        return f"/clips/{rel}"
    # Fallback: try target mount
    rel = osp.basename(ap)
    return f"/videos/target/{rel}"


def _assign_clip_metadata(entry: dict, clip_path: str) -> None:
    """Update a search result entry so the clip under CLIP_DIR is displayed."""
    clip_abs = osp.abspath(clip_path)
    clip_url = _rel_url(clip_path)
    clip_name = osp.basename(clip_path)
    entry["clip_url"] = clip_url
    entry["clip_abs"] = clip_abs
    entry["clip_path"] = clip_path
    entry["clip_name"] = clip_name
    entry["display_name"] = clip_name
    entry.setdefault("name", clip_name)


@app.get("/", response_class=HTMLResponse)
@debug_log
async def index(request: Request):
    """Render the search UI template."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "target_root": TARGET_ROOT,
            "reference_root": REFERENCE_ROOT,
            "video_root": TARGET_ROOT,
        },
    )


@app.get("/api/videos")
@debug_log
async def list_videos():
    """Expose the list of available target and clip videos."""
    targets = [
        {
            "name": name,
            "path": osp.abspath(osp.join(TARGET_ROOT, name)),
            "url": _rel_url(osp.join(TARGET_ROOT, name)),
            "source": "target",
        }
        for name in _list_videos(TARGET_ROOT)
    ]

    clips = []
    if osp.isdir(CLIP_DIR):
        clips = [
            {
                "name": name,
                "path": osp.abspath(osp.join(CLIP_DIR, name)),
                "url": _rel_url(osp.join(CLIP_DIR, name)),
                "source": "clip",
            }
            for name in _list_videos(CLIP_DIR)
        ]

    return {
        "root": REFERENCE_ROOT,
        "target_root": TARGET_ROOT,
        "reference_root": REFERENCE_ROOT,
        "clip_root": CLIP_DIR,
        "videos": targets,
        "clips": clips,
    }


@app.post("/api/search")
@debug_log
async def search(payload: dict):
    """Run similarity search for the requested target against candidate videos."""
    target = payload.get("target")
    candidates_dir = payload.get("candidates_dir")
    device = payload.get("device", "AUTO")
    topk = int(payload.get("topk", 5))
    frame_stride = int(payload.get("frame_stride", 5))
    swing_only = bool(payload.get("swing_only", True))
    swing_seconds = payload.get("swing_seconds", 5)
    recompute = bool(payload.get("recompute", False))

    logger.debug("Search request: target=%s candidates_dir=%s device=%s topk=%s stride=%s",
                 target, candidates_dir, device, topk, frame_stride)

    if not target:
        raise HTTPException(status_code=400, detail="target is required")

    # Resolve target path relative to TARGET_ROOT if not absolute
    tgt_path = target
    if not osp.isabs(tgt_path):
        tgt_path = osp.join(TARGET_ROOT, tgt_path)

    # Resolve candidates relative to CANDIDATE_ROOT if not provided/absolute
    if not candidates_dir:
        cand_dir = REFERENCE_ROOT
    else:
        cand_dir = candidates_dir
        if not osp.isabs(cand_dir):
            cand_dir = osp.join(REFERENCE_ROOT, cand_dir)

    logger.debug("Resolved paths: tgt_path=%s cand_dir=%s", tgt_path, cand_dir)

    if not osp.isfile(tgt_path):
        raise HTTPException(status_code=404, detail=f"Target not found: {tgt_path}")
    if not osp.isdir(cand_dir) and not osp.isfile(cand_dir):
        raise HTTPException(status_code=404, detail=f"Candidates not found: {cand_dir}")

    model = _ensure_model()

    # Helper to run feature extraction pipeline with a specific device
    def _run_with_device(dev: str):
        # Caches
        swing_cache_dir = "features_cache_swing"
        full_cache_dir = "features_cache"
        clip_cache_dir = "features_cache_clip"
        # If recompute requested, remove any existing caches for the target
        if recompute:
            try:
                for d in (full_cache_dir, swing_cache_dir, clip_cache_dir):
                    base = osp.splitext(osp.basename(tgt_path))[0]
                    cpath = osp.join(d, f"{base}.npz")
                    if osp.exists(cpath):
                        os.remove(cpath)
            except Exception:
                pass
        # 1) Target: ensure a clip exists first, then compute features on the clip
        tgt_window = None
        # Try read swing window from cache
        try:
            meta = load_feature_meta(swing_cache_dir, tgt_path)
            ws = meta.get("window_start_sec")
            we = meta.get("window_end_sec")
            if ws is not None and we is not None:
                tgt_window = (float(ws), float(we))
        except Exception:
            pass
        if tgt_window is None or recompute:
            info = extract_video_features(
                tgt_path,
                model,
                frame_stride=frame_stride,
                device=dev,
                swing_only=True,
                swing_seconds=swing_seconds,
            )
            ws = float(info.get("window_start_sec", -1.0)) if isinstance(info, dict) else -1.0
            we = float(info.get("window_end_sec", -1.0)) if isinstance(info, dict) else -1.0
            if ws >= 0 and we >= 0:
                tgt_window = (ws, we)
            save_feature_cache(
                swing_cache_dir,
                tgt_path,
                info["vector"],
                window_start_sec=(ws if ws >= 0 else None),
                window_end_sec=(we if we >= 0 else None),
                frame_stride=frame_stride,
            )

        # Create or find target clip
        tgt_clip_path = None
        if tgt_window is not None:
            tws, twe = tgt_window
            tgt_clip_path = make_video_clip(
                tgt_path,
                float(tws),
                float(twe),
                CLIP_DIR,
                basename=osp.splitext(osp.basename(tgt_path))[0],
            )
        # Compute target vector from the clip (preferred)
        tgt_vec = None
        if tgt_clip_path and osp.exists(tgt_clip_path):
            tgt_vec = None if recompute else load_feature_cache(clip_cache_dir, tgt_clip_path)
            if tgt_vec is None:
                info_clip = extract_video_features(
                    tgt_clip_path,
                    model,
                    frame_stride=frame_stride,
                    device=dev,
                    swing_only=False,
                    swing_seconds=swing_seconds,
                )
                tgt_vec = info_clip["vector"]
                save_feature_cache(clip_cache_dir, tgt_clip_path, tgt_vec, frame_stride=frame_stride)
        # Fallback to full video vector if clip failed
        if tgt_vec is None:
            tgt_vec = None if recompute else load_feature_cache(full_cache_dir, tgt_path)
            if tgt_vec is None:
                info_full = extract_video_features(
                    tgt_path,
                    model,
                    frame_stride=frame_stride,
                    device=dev,
                    swing_only=True,
                    swing_seconds=swing_seconds,
                )
                tgt_vec = info_full["vector"]
                save_feature_cache(full_cache_dir, tgt_path, tgt_vec)

        # Candidates
        cand_paths: List[Tuple[str, object]] = []
        if osp.isdir(cand_dir):
            for f in sorted(os.listdir(cand_dir)):
                if not _is_video(f):
                    continue
                p = osp.join(cand_dir, f)
                if osp.abspath(p) == osp.abspath(tgt_path):
                    continue
                cand_paths.append((p, None))
        elif osp.isfile(cand_dir):
            if osp.abspath(cand_dir) != osp.abspath(tgt_path):
                cand_paths.append((cand_dir, None))

        logger.debug("Candidates discovered: %d", len(cand_paths))

        # Compute features for candidates (create clips first, then compare)
        cand_vecs: List[Tuple[str, object]] = []
        clip_pool: List[dict] = []
        used_clip_paths = set()
        for p, _ in cand_paths:
            # Ensure swing window and clip for candidate
            c_window = None
            try:
                meta = load_feature_meta(swing_cache_dir, p)
                ws = meta.get("window_start_sec")
                we = meta.get("window_end_sec")
                if ws is not None and we is not None:
                    c_window = (float(ws), float(we))
            except Exception:
                pass
            if c_window is None or recompute:
                info = extract_video_features(
                    p,
                    model,
                    frame_stride=frame_stride,
                    device=dev,
                    swing_only=True,
                    swing_seconds=swing_seconds,
                )
                ws = float(info.get("window_start_sec", -1.0)) if isinstance(info, dict) else -1.0
                we = float(info.get("window_end_sec", -1.0)) if isinstance(info, dict) else -1.0
                if ws >= 0 and we >= 0:
                    c_window = (ws, we)
                save_feature_cache(
                    swing_cache_dir,
                    p,
                    info["vector"],
                    window_start_sec=(ws if ws >= 0 else None),
                    window_end_sec=(we if we >= 0 else None),
                    frame_stride=frame_stride,
                )
            clip_path = None
            if c_window is not None:
                cws, cwe = c_window
                clip_path = make_video_clip(p, float(cws), float(cwe), CLIP_DIR, basename=osp.splitext(osp.basename(p))[0])
            # Compute vector on clip
            vec_c = None
            if clip_path and osp.exists(clip_path):
                vec_c = None if recompute else load_feature_cache(clip_cache_dir, clip_path)
                if vec_c is None:
                    info_c = extract_video_features(
                        clip_path,
                        model,
                        frame_stride=frame_stride,
                        device=dev,
                        swing_only=False,
                        swing_seconds=swing_seconds,
                    )
                    vec_c = info_c["vector"]
                    save_feature_cache(clip_cache_dir, clip_path, vec_c, frame_stride=frame_stride)
            # Fallback: compute on full video window if clip missing
            if vec_c is None:
                info_f = extract_video_features(
                    p,
                    model,
                    frame_stride=frame_stride,
                    device=dev,
                    swing_only=True,
                    swing_seconds=swing_seconds,
                )
                vec_c = info_f["vector"]
            cand_vecs.append((clip_path or p, vec_c))

        ranked = rank_similar(tgt_vec, cand_vecs)[:topk]
        logger.debug("Ranking complete: returned=%d", len(ranked))
        results = []
        for (p, score) in ranked:
            item = {
                "path": p,
                "path_abs": osp.abspath(p),
                "name": osp.basename(p),
                "display_name": osp.basename(p),
                "score": float(score),
                "url": _rel_url(p),
                "original_path": p,
                "source": "reference",
            }
            # Prefer the generated clip for display
            try:
                base = osp.splitext(osp.basename(p))[0]
                pattern = osp.join(CLIP_DIR, f"{base}_clip*.mp4")
                matches = sorted(glob.glob(pattern))
                if matches:
                    _assign_clip_metadata(item, matches[0])
            except Exception:
                pass
            # Fallback: if a pre-generated clip exists in the clip directory, use it
            if "clip_url" not in item:
                base = osp.splitext(osp.basename(p))[0]
                pattern = osp.join(CLIP_DIR, f"{base}_clip*.mp4")
                matches = sorted(glob.glob(pattern))
                if matches:
                    pre_clip = matches[0]
                    _assign_clip_metadata(item, pre_clip)
            if item.get("clip_abs"):
                used_clip_paths.add(item["clip_abs"])
            results.append(item)

        target_entry = {
            "path": tgt_path,
            "path_abs": osp.abspath(tgt_path),
            "name": osp.basename(tgt_path),
            "display_name": osp.basename(tgt_path),
            "url": _rel_url(tgt_path),
            "original_path": tgt_path,
            "source": "target",
        }
        if tgt_window:
            tws, twe = tgt_window
            target_entry["start"], target_entry["end"] = float(tws), float(twe)
            if tgt_clip_path and osp.exists(tgt_clip_path):
                _assign_clip_metadata(target_entry, tgt_clip_path)
                if target_entry.get("clip_abs"):
                    used_clip_paths.add(target_entry["clip_abs"])
        # Fallback: use existing clip if present even when swing_only is false
        if "clip_url" not in target_entry:
            base = osp.splitext(osp.basename(tgt_path))[0]
            pattern = osp.join(CLIP_DIR, f"{base}_clip*.mp4")
            matches = sorted(glob.glob(pattern))
            if matches:
                pre_clip = matches[0]
                _assign_clip_metadata(target_entry, pre_clip)
                if target_entry.get("clip_abs"):
                    used_clip_paths.add(target_entry["clip_abs"])

        if osp.isdir(CLIP_DIR):
            for name in _list_videos(CLIP_DIR):
                clip_full = osp.join(CLIP_DIR, name)
                clip_abs = osp.abspath(clip_full)
                if clip_abs in used_clip_paths:
                    continue
                clip_pool.append({
                    "path": clip_full,
                    "path_abs": clip_abs,
                    "name": name,
                    "display_name": name,
                    "url": _rel_url(clip_full),
                    "clip_url": _rel_url(clip_full),
                    "clip_abs": clip_abs,
                    "clip_path": clip_full,
                    "clip_name": name,
                    "source": "clip",
                })

        return {"used_device": dev, "target": target_entry, "results": results, "clip_pool": clip_pool}

    # Try requested device; on failure, retry with CPU
    try:
        return _run_with_device(device)
    except Exception as e:
        logger.exception("Search failed on device %s, retrying on CPU: %s", device, e)
        try:
            return _run_with_device("CPU")
        except Exception:
            # Give up; propagate original error
            raise


if __name__ == "__main__":
    uvicorn.run("similarity_check.web_api:app", host="127.0.0.1", port=8000, reload=False)

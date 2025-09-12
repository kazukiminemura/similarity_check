import os
import os.path as osp
import logging
import functools
import inspect
from typing import List, Tuple, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from similarity_check.features import (
    load_model,
    extract_video_features,
    load_feature_cache,
    save_feature_cache,
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
    # If env is set and absolute, use as-is. If relative or not set, resolve from BASE_DIR.
    val = os.environ.get(env_name)
    if val:
        return val if osp.isabs(val) else osp.abspath(osp.join(BASE_DIR, val))
    return osp.abspath(osp.join(BASE_DIR, default_rel))

TARGET_ROOT = _resolve_root("TARGET_ROOT", "target")
REFERENCE_ROOT = _resolve_root("REFERENCE_ROOT", "reference")
STATIC_DIR = osp.join(osp.dirname(__file__), "static")
TEMPLATE_DIR = osp.join(osp.dirname(__file__), "templates")
CLIP_DIR = osp.join(osp.dirname(__file__), "_clips")

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


@debug_log
def _ensure_model():
    global _model
    if _model is None:
        _model = load_model("yolov8n-pose.pt")
    return _model


@debug_log
def _is_video(fname: str) -> bool:
    f = fname.lower()
    return f.endswith((".mp4", ".mov", ".avi", ".mkv", ".m4v"))


@debug_log
def _list_videos(root: str) -> List[str]:
    if not osp.isdir(root):
        return []
    return sorted([f for f in os.listdir(root) if _is_video(f)])


@debug_log
def _rel_url(path: str) -> str:
    # Convert absolute/relative file path to a mounted static URL
    ap = osp.abspath(path)
    if osp.isdir(TARGET_ROOT) and osp.commonpath([ap, TARGET_ROOT]) == TARGET_ROOT:
        rel = osp.relpath(ap, TARGET_ROOT).replace("\\", "/")
        return f"/videos/target/{rel}"
    if osp.isdir(REFERENCE_ROOT) and osp.commonpath([ap, REFERENCE_ROOT]) == REFERENCE_ROOT:
        rel = osp.relpath(ap, REFERENCE_ROOT).replace("\\", "/")
        return f"/videos/reference/{rel}"
    # Fallback: try target mount
    rel = osp.basename(ap)
    return f"/videos/target/{rel}"


@app.get("/", response_class=HTMLResponse)
@debug_log
async def index(request: Request):
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
    items = _list_videos(TARGET_ROOT)
    return {"root": REFERENCE_ROOT, "target_root": TARGET_ROOT, "reference_root": REFERENCE_ROOT, "videos": items}


@app.post("/api/search")
@debug_log
async def search(payload: dict):
    target = payload.get("target")
    candidates_dir = payload.get("candidates_dir")
    device = payload.get("device", "AUTO")
    topk = int(payload.get("topk", 5))
    frame_stride = int(payload.get("frame_stride", 5))
    swing_only = bool(payload.get("swing_only", True))
    swing_seconds = payload.get("swing_seconds", 2.5)
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
        # Target features (with cache) - swing-only cache dir
        cache_dir = "features_cache_swing" if swing_only else "features_cache"
        # If recompute requested, remove any existing caches for the target
        if recompute:
            try:
                for d in ("features_cache", "features_cache_swing"):
                    base = osp.splitext(osp.basename(tgt_path))[0]
                    cpath = osp.join(d, f"{base}.npz")
                    if osp.exists(cpath):
                        os.remove(cpath)
            except Exception:
                pass
        vec = None if recompute else load_feature_cache(cache_dir, tgt_path)
        tgt_window = None
        if vec is None:
            info = extract_video_features(
                tgt_path,
                model,
                frame_stride=frame_stride,
                device=dev,
                swing_only=swing_only,
                swing_seconds=swing_seconds,
            )
            vec = info["vector"]
            save_feature_cache(cache_dir, tgt_path, vec, window_start_sec=(float(info.get("window_start_sec", -1.0)) if isinstance(info, dict) else None), window_end_sec=(float(info.get("window_end_sec", -1.0)) if isinstance(info, dict) else None), frame_stride=frame_stride)

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

        # Compute features for candidates
        cand_vecs: List[Tuple[str, object]] = []
        cand_windows = {}
        for p, _ in cand_paths:
            if recompute:
                try:
                    for d in ("features_cache", "features_cache_swing"):
                        base = osp.splitext(osp.basename(p))[0]
                        cpath = osp.join(d, f"{base}.npz")
                        if osp.exists(cpath):
                            os.remove(cpath)
                except Exception:
                    pass
            v = None if recompute else load_feature_cache(cache_dir, p)
            if v is None:
                info = extract_video_features(
                    p,
                    model,
                    frame_stride=frame_stride,
                    device=dev,
                    swing_only=swing_only,
                    swing_seconds=swing_seconds,
                )
                v = info["vector"]
                ws = float(info.get("window_start_sec", -1.0)) if isinstance(info, dict) else -1.0
                we = float(info.get("window_end_sec", -1.0)) if isinstance(info, dict) else -1.0
                save_feature_cache(cache_dir, p, v, window_start_sec=(ws if ws >= 0 else None), window_end_sec=(we if ws >= 0 else None), frame_stride=frame_stride)
                if ws >= 0 and we >= 0:
                    cand_windows[p] = (ws, we)
            else:
                from similarity_check.features import load_feature_meta as _lfm
                meta = _lfm(cache_dir, p)
                ws = meta.get("window_start_sec")
                we = meta.get("window_end_sec")
                if ws is not None and we is not None:
                    cand_windows[p] = (ws, we)
            cand_vecs.append((p, v))

        ranked = rank_similar(vec, cand_vecs)[:topk]
        logger.debug("Ranking complete: returned=%d", len(ranked))
        results = []
        for (p, score) in ranked:
            item = {"path": p, "name": osp.basename(p), "score": float(score), "url": _rel_url(p)}
            if swing_only:
                # try read window meta from cache if available
                try:
                    from similarity_check.features import load_feature_meta as _lfm
                    meta = _lfm(cache_dir, p)
                    ws = meta.get("window_start_sec")
                    we = meta.get("window_end_sec")
                    if ws is not None and we is not None:
                        item["start"], item["end"] = float(ws), float(we)
                        clip_path = make_video_clip(p, float(ws), float(we), CLIP_DIR, basename=osp.splitext(osp.basename(p))[0])
                        if clip_path and osp.exists(clip_path):
                            item["clip_url"] = "/clips/" + osp.basename(clip_path)
                            # Use clip filename (with _clip suffix) as display name
                            item["name"] = osp.basename(clip_path)
                except Exception:
                    pass
            # Fallback: if a pre-generated clip exists in _clips, use it
            if "clip_url" not in item:
                base = osp.splitext(osp.basename(p))[0]
                pre_clip = osp.join(CLIP_DIR, f"{base}_clip.mp4")
                if osp.exists(pre_clip):
                    item["clip_url"] = "/clips/" + osp.basename(pre_clip)
                    item["name"] = osp.basename(pre_clip)
            results.append(item)

        target_entry = {"path": tgt_path, "name": osp.basename(tgt_path), "url": _rel_url(tgt_path)}
        if swing_only and tgt_window:
            tws, twe = tgt_window
            target_entry["start"], target_entry["end"] = float(tws), float(twe)
            clip_path = make_video_clip(tgt_path, float(tws), float(twe), CLIP_DIR, basename=osp.splitext(osp.basename(tgt_path))[0])
            if clip_path and osp.exists(clip_path):
                target_entry["clip_url"] = "/clips/" + osp.basename(clip_path)
                # Use clip filename (with _clip suffix) as display name
                target_entry["name"] = osp.basename(clip_path)
        # Fallback: use existing clip if present even when swing_only is false
        if "clip_url" not in target_entry:
            base = osp.splitext(osp.basename(tgt_path))[0]
            pre_clip = osp.join(CLIP_DIR, f"{base}_clip.mp4")
            if osp.exists(pre_clip):
                target_entry["clip_url"] = "/clips/" + osp.basename(pre_clip)
                target_entry["name"] = osp.basename(pre_clip)

        return {"used_device": dev, "target": target_entry, "results": results}

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
    import uvicorn

    uvicorn.run("similarity_check.web_api:app", host="127.0.0.1", port=8000, reload=False)

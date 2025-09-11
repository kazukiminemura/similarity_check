import os
import os.path as osp
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


VIDEO_ROOT = osp.abspath(os.environ.get("VIDEO_ROOT", "data"))
STATIC_DIR = osp.join(osp.dirname(__file__), "static")
TEMPLATE_DIR = osp.join(osp.dirname(__file__), "templates")

app = FastAPI(title="Pose Similarity (YOLOv8-Pose) API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if osp.isdir(VIDEO_ROOT):
    app.mount("/videos", StaticFiles(directory=VIDEO_ROOT), name="videos")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

_model = None


def _ensure_model():
    global _model
    if _model is None:
        _model = load_model("yolov8n-pose.pt")
    return _model


def _is_video(fname: str) -> bool:
    f = fname.lower()
    return f.endswith((".mp4", ".mov", ".avi", ".mkv", ".m4v"))


def _list_videos(root: str) -> List[str]:
    if not osp.isdir(root):
        return []
    return sorted([f for f in os.listdir(root) if _is_video(f)])


def _rel_url(path: str) -> str:
    # Convert absolute/relative file path to /videos/<rel>
    rel = osp.relpath(osp.abspath(path), VIDEO_ROOT)
    rel = rel.replace("\\", "/")
    return f"/videos/{rel}"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "video_root": VIDEO_ROOT})


@app.get("/api/videos")
async def list_videos():
    items = _list_videos(VIDEO_ROOT)
    return {"root": VIDEO_ROOT, "videos": items}


@app.post("/api/search")
async def search(payload: dict):
    target = payload.get("target")
    candidates_dir = payload.get("candidates_dir") or VIDEO_ROOT
    topk = int(payload.get("topk", 5))
    frame_stride = int(payload.get("frame_stride", 5))

    if not target:
        raise HTTPException(status_code=400, detail="target is required")

    # Resolve paths relative to VIDEO_ROOT if not absolute
    tgt_path = target
    if not osp.isabs(tgt_path):
        tgt_path = osp.join(VIDEO_ROOT, tgt_path)

    cand_dir = candidates_dir
    if not osp.isabs(cand_dir):
        cand_dir = osp.join(VIDEO_ROOT, cand_dir)

    if not osp.isfile(tgt_path):
        raise HTTPException(status_code=404, detail=f"Target not found: {tgt_path}")
    if not osp.isdir(cand_dir) and not osp.isfile(cand_dir):
        raise HTTPException(status_code=404, detail=f"Candidates not found: {cand_dir}")

    model = _ensure_model()

    # Target features (with cache)
    vec = load_feature_cache("features_cache", tgt_path)
    if vec is None:
        info = extract_video_features(tgt_path, model, frame_stride=frame_stride)
        vec = info["vector"]
        save_feature_cache("features_cache", tgt_path, vec)

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

    # Compute features for candidates
    cand_vecs: List[Tuple[str, object]] = []
    for p, _ in cand_paths:
        v = load_feature_cache("features_cache", p)
        if v is None:
            info = extract_video_features(p, model, frame_stride=frame_stride)
            v = info["vector"]
            save_feature_cache("features_cache", p, v)
        cand_vecs.append((p, v))

    ranked = rank_similar(vec, cand_vecs)[:topk]
    results = [
        {"path": p, "name": osp.basename(p), "score": float(score), "url": _rel_url(p)}
        for (p, score) in ranked
    ]
    return {
        "target": {"path": tgt_path, "name": osp.basename(tgt_path), "url": _rel_url(tgt_path)},
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("similarity_check.web_api:app", host="127.0.0.1", port=8000, reload=False)


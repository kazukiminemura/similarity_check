import argparse
import glob
import os
import os.path as osp
from typing import List, Tuple

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from similarity_check.features import (
    load_model,
    extract_video_features,
    save_feature_cache,
    load_feature_cache,
    make_video_thumbnail_with_pose,
)
from similarity_check.similarity import rank_similar


def find_videos(path: str) -> List[str]:
    if osp.isdir(path):
        vids: List[str] = []
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.m4v"):
            vids.extend(glob.glob(osp.join(path, ext)))
        return sorted(vids)
    if osp.isfile(path):
        return [path]
    return []


def save_montage(thumbnails: List[Tuple[str, float, np.ndarray]], out_path: str = "results_montage.jpg") -> str:
    if not thumbnails:
        return out_path
    # make a vertical stack, add score text
    rows = []
    for path, score, img in thumbnails:
        if img is None:
            continue
        h, w = img.shape[:2]
        canvas = img.copy()
        cv2.rectangle(canvas, (0, 0), (w, 30), (0, 0, 0), -1)
        text = f"{osp.basename(path)}  score={score:.3f}"
        cv2.putText(canvas, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(canvas)
    if not rows:
        return out_path
    # normalize widths
    maxw = max(r.shape[1] for r in rows)
    norm_rows = []
    for r in rows:
        if r.shape[1] != maxw:
            scale = maxw / r.shape[1]
            r = cv2.resize(r, (maxw, int(r.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        norm_rows.append(r)
    montage = np.vstack(norm_rows)
    cv2.imwrite(out_path, montage)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Find similar videos by pose (YOLOv8-Pose)")
    parser.add_argument("--target", required=True, help="Target video path")
    parser.add_argument("--candidates", required=True, help="Folder or video file(s) to search")
    parser.add_argument("--topk", type=int, default=5, help="Top-K to show")
    parser.add_argument("--frame-stride", type=int, default=5, help="Sample every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit processed frames per video")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Ultralytics model name or path")
    parser.add_argument("--cache-dir", default="features_cache", help="Feature cache directory")
    parser.add_argument("--no-visualize", action="store_true", help="Skip thumbnail creation")
    args = parser.parse_args()

    console = Console()

    if not osp.exists(args.target):
        console.print(f"[red]Target not found:[/red] {args.target}")
        raise SystemExit(1)

    cand_paths = find_videos(args.candidates)
    cand_paths = [p for p in cand_paths if osp.abspath(p) != osp.abspath(args.target)]
    if not cand_paths:
        console.print(f"[red]No candidate videos found in:[/red] {args.candidates}")
        raise SystemExit(1)

    console.print(f"Loading model: {args.model}")
    model = load_model(args.model)

    # Target features
    tgt_vec = load_feature_cache(args.cache_dir, args.target)
    if tgt_vec is None:
        console.print("Extracting target features...")
        tgt = extract_video_features(args.target, model, frame_stride=args.frame_stride, max_frames=args.max_frames)
        tgt_vec = tgt["vector"]
        save_feature_cache(args.cache_dir, args.target, tgt_vec)
    else:
        console.print("Loaded target features from cache.")

    # Candidate features
    candidates: List[Tuple[str, np.ndarray]] = []
    for p in cand_paths:
        vec = load_feature_cache(args.cache_dir, p)
        if vec is None:
            info = extract_video_features(p, model, frame_stride=args.frame_stride, max_frames=args.max_frames)
            vec = info["vector"]
            save_feature_cache(args.cache_dir, p, vec)
        candidates.append((p, vec))

    ranked = rank_similar(tgt_vec, candidates)

    table = Table(title="Similar Videos (by Pose)")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Video")

    topk = max(1, min(args.topk, len(ranked)))
    thumbs: List[Tuple[str, float, np.ndarray]] = []
    for i, (path, score) in enumerate(ranked[:topk], 1):
        table.add_row(str(i), f"{score:.3f}", path)
        if not args.no_visualize:
            try:
                img = make_video_thumbnail_with_pose(path, model)
            except Exception:
                img = None
            if img is not None:
                thumbs.append((path, score, img))

    console.print(table)

    if thumbs and not args.no_visualize:
        out = save_montage(thumbs, out_path="results_montage.jpg")
        console.print(f"Saved thumbnails montage: {out}")


if __name__ == "__main__":
    main()

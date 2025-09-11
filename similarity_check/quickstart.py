import os
import os.path as osp
from typing import List, Tuple

import numpy as np

from similarity_check.features import (
    load_model,
    extract_video_features,
    load_feature_cache,
    save_feature_cache,
    make_video_thumbnail_with_pose,
)
from similarity_check.similarity import rank_similar


def find_videos(path: str) -> List[str]:
    if osp.isdir(path):
        exts = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
        return sorted(
            [osp.join(path, f) for f in os.listdir(path) if f.lower().endswith(exts)]
        )
    return [path] if osp.isfile(path) else []


def prompt(msg: str, default: str = "") -> str:
    text = input(f"{msg} [{default}]: ").strip()
    return text or default


def main():
    print("\n=== Pose Similarity Quickstart (YOLOv8-Pose) ===\n")
    default_target = "data/front_target1.mp4" if osp.exists("data/front_target1.mp4") else ""
    default_cands = "data" if osp.isdir("data") else ""

    target = prompt("Target video path", default_target)
    while not osp.isfile(target):
        print("[!] File not found. Try again.")
        target = prompt("Target video path", default_target)

    candidates = prompt("Candidates folder or video path", default_cands)
    while not (osp.isdir(candidates) or osp.isfile(candidates)):
        print("[!] Folder/file not found. Try again.")
        candidates = prompt("Candidates folder or video path", default_cands)

    topk = prompt("Top-K results", "5")
    frame_stride = prompt("Frame stride (sample every Nth frame)", "5")
    model_name = prompt("Model (Ultralytics) name/path", "yolov8n-pose.pt")
    cache_dir = prompt("Feature cache directory", "features_cache")

    try:
        topk = max(1, int(topk))
    except Exception:
        topk = 5
    try:
        frame_stride = max(1, int(frame_stride))
    except Exception:
        frame_stride = 5

    print(f"\n[+] Loading model: {model_name}")
    model = load_model(model_name)

    # Target features (cache)
    vec = load_feature_cache(cache_dir, target)
    if vec is None:
        print("[+] Extracting target features...")
        info = extract_video_features(target, model, frame_stride=frame_stride)
        vec = info["vector"]
        save_feature_cache(cache_dir, target, vec)
    else:
        print("[+] Loaded target features from cache.")

    # Candidates
    cand_paths = find_videos(candidates)
    cand_paths = [p for p in cand_paths if osp.abspath(p) != osp.abspath(target)]
    if not cand_paths:
        print("[!] No candidate videos found.")
        return

    cand_vecs: List[Tuple[str, np.ndarray]] = []
    for p in cand_paths:
        v = load_feature_cache(cache_dir, p)
        if v is None:
            print(f"[+] Extracting: {osp.basename(p)}")
            info = extract_video_features(p, model, frame_stride=frame_stride)
            v = info["vector"]
            save_feature_cache(cache_dir, p, v)
        cand_vecs.append((p, v))

    ranked = rank_similar(vec, cand_vecs)[:topk]

    print("\n=== Results ===")
    for i, (path, score) in enumerate(ranked, 1):
        print(f"{i:2d}. {osp.basename(path)}\t score={score:.3f}")

    try:
        print("[+] Generating thumbnails montage: results_montage.jpg")
        from similarity_check.cli import save_montage, make_video_thumbnail_with_pose  # reuse
        thumbs = []
        for path, score in ranked:
            img = make_video_thumbnail_with_pose(path, model)
            if img is not None:
                thumbs.append((path, score, img))
        if thumbs:
            out = save_montage(thumbs, out_path="results_montage.jpg")
            print(f"[+] Saved: {out}")
    except Exception:
        pass

    print("\nDone.\n")


if __name__ == "__main__":
    main()


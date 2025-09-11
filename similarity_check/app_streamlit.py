import os
import os.path as osp
from typing import List, Tuple

import numpy as np
import streamlit as st

from .features import load_model, extract_video_features, load_feature_cache, save_feature_cache, make_video_thumbnail_with_pose
from .similarity import rank_similar


def find_videos(path: str) -> List[str]:
    if osp.isdir(path):
        vids: List[str] = []
        for ext in (".mp4", ".mov", ".avi", ".mkv", ".m4v"):
            vids.extend([osp.join(path, f) for f in os.listdir(path) if f.lower().endswith(ext)])
        return sorted(vids)
    if osp.isfile(path):
        return [path]
    return []


def main():
    st.set_page_config(page_title="Pose Similarity (YOLOv8)", layout="wide")
    st.title("動画のポーズ類似検索 (YOLOv8-Pose)")

    target_path = st.text_input("対象動画のパス", "")
    candidates_dir = st.text_input("候補動画フォルダのパス", "")
    topk = st.slider("表示数", 1, 20, 5)
    frame_stride = st.number_input("フレーム間引き (Nフレーム毎)", min_value=1, max_value=30, value=5)
    model_name = st.text_input("モデル名/パス", "yolov8n-pose.pt")
    cache_dir = st.text_input("キャッシュ保存先", "features_cache")

    if st.button("検索実行", disabled=not (osp.exists(target_path) and osp.exists(candidates_dir))):
        with st.spinner("モデル読込中..."):
            model = load_model(model_name)

        # Target features
        vec = load_feature_cache(cache_dir, target_path)
        if vec is None:
            with st.spinner("対象動画の特徴抽出中..."):
                info = extract_video_features(target_path, model, frame_stride=frame_stride)
                vec = info["vector"]
                save_feature_cache(cache_dir, target_path, vec)

        # Candidates
        cand_paths = find_videos(candidates_dir)
        cand_paths = [p for p in cand_paths if osp.abspath(p) != osp.abspath(target_path)]
        cand_vecs: List[Tuple[str, np.ndarray]] = []
        for p in cand_paths:
            v = load_feature_cache(cache_dir, p)
            if v is None:
                with st.spinner(f"特徴抽出: {osp.basename(p)}"):
                    info = extract_video_features(p, model, frame_stride=frame_stride)
                    v = info["vector"]
                    save_feature_cache(cache_dir, p, v)
            cand_vecs.append((p, v))

        ranked = rank_similar(vec, cand_vecs)[:topk]

        st.subheader("結果")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**対象動画**")
            try:
                st.video(target_path)
            except Exception:
                st.text(osp.basename(target_path))
        with cols[1]:
            st.markdown("**上位候補**")
            for i, (path, score) in enumerate(ranked, 1):
                st.write(f"{i}. {osp.basename(path)}  (score={score:.3f})")
                try:
                    img = make_video_thumbnail_with_pose(path, model)
                    if img is not None:
                        st.image(img[:, :, ::-1], caption=osp.basename(path))
                except Exception:
                    pass


if __name__ == "__main__":
    main()


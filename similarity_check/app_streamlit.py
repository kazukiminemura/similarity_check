import os
import os.path as osp
from typing import List, Tuple

import numpy as np
import streamlit as st

from similarity_check.features import load_model, extract_video_features, load_feature_cache, save_feature_cache, make_video_thumbnail_with_pose
from similarity_check.similarity import rank_similar
from similarity_check.video_utils import make_sync_grid_video


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
    topk = st.slider("表示数 (上位K)", 1, 20, 5)
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
        # レイアウト: 2行x3列グリッドに動画プレイヤーを配置（左上が対象）
        st.markdown("### プレビュー（個別再生）")
        grid_cols = st.columns(3)
        try:
            with grid_cols[0]:
                st.markdown("**左上: 対象動画**")
                st.video(target_path)
        except Exception:
            with grid_cols[0]:
                st.text(osp.basename(target_path))

        # 上位5件を埋める
        slots = [None] * 5
        for i, (path, score) in enumerate(ranked[:5], 0):
            slots[i] = (path, score)
        # 配置: [1]=top1, [2]=top2 右上, [3]=top3 左下, [4]=top4 中下, [5]=top5 右下
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]  # (row, col)
        for idx, slot in enumerate(slots):
            if slot is None:
                continue
            path, score = slot
            row, col = positions[idx]
            if row == 0:
                col_container = grid_cols[col]
            else:
                # second row
                if idx == 2:
                    grid_cols2 = st.columns(3)
                col_container = grid_cols2[col]
            with col_container:
                st.markdown(f"**Top {idx+1}: {osp.basename(path)} (score={score:.3f})**")
                try:
                    st.video(path)
                except Exception:
                    st.text(osp.basename(path))

        # 同期再生（2x3グリッドで1本の動画を生成して再生）
        st.markdown("### 同期再生（グリッド動画を生成）")
        max_seconds = st.number_input("最大再生秒数（空=自動）", min_value=1, max_value=120, value=20)
        if st.button("同期グリッド動画を生成・再生"):
            with st.spinner("同期グリッド動画を生成中..."):
                top_paths = [p for (p, _) in ranked[:5]]
                out = make_sync_grid_video(
                    target_path,
                    top_paths,
                    out_path="results_sync_grid.mp4",
                    rows=2,
                    cols=3,
                    cell_size=(400, 225),  # 16:9 cells
                    max_seconds=float(max_seconds) if max_seconds else None,
                )
            st.success(f"生成しました: {out}")
            st.video(out)


if __name__ == "__main__":
    main()

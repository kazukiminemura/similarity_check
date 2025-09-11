# Video Similarity by Pose (YOLOv8-Pose)

複数の動画から、対象動画にポーズ（骨格）が近い動画を探して表示するツールです。Ultralytics YOLOv8-Pose でキーポイントを抽出し、動画ごとの特徴量を作ってコサイン類似度でランキングします。

## いちばん簡単（Windows）
- Streamlit のUIで使う（自動で仮想環境作成＆依存導入）
  - `run_ui.bat` をダブルクリック
  - ブラウザが開いたら、対象・候補のパスを入力して「検索実行」
- コマンドラインで使う（自動で仮想環境作成＆依存導入）
  - 例: `run_cli.bat data\front_target1.mp4 data 5 5`

どちらも初回は YOLO の重み `yolov8n-pose.pt` を自動ダウンロードします（ネット接続が必要）。

## 手動セットアップ（任意）
1. Python 3.9+（Windows可）を用意
2. 依存をインストール

```bash
pip install -r requirements.txt
# UIも使う場合:
pip install -r requirements-ui.txt
```

## ルートから実行（Python）
- リポジトリ直下でそのまま実行できます。

```bash
python main.py --target path/to/target.mp4 --candidates path/to/video_dir --topk 5 --frame-stride 5 --model yolov8n-pose.pt
```

同等の実行（好みで選択）:
- `python -m similarity_check.cli ...`
- `python -m similarity_check ...`（パッケージ実行）

## CLI（詳細）
- 実行例:
```bash
python -m similarity_check.cli \
  --target path/to/target.mp4 \
  --candidates path/to/video_dir \
  --topk 5 \
  --frame-stride 5 \
  --model yolov8n-pose.pt
```
- 出力:
  - 類似度ランキングをコンソール表示
  - `results_montage.jpg` に上位サムネイル（骨格オーバーレイ）
  - `features_cache/*.npz` に特徴量キャッシュ

## 対話式クイックスタート（もっと簡単）
質問に答えるだけで実行します。

```bash
python -m similarity_check.quickstart
```

## Streamlit UI
```bash
streamlit run similarity_check/app_streamlit.py
```

## 仕組み（概要）
- フレーム毎に最も信頼度の高い人物の17点キーポイントを取得
- 位置・スケール不変になるよう正規化（重心平行移動＋スケーリング）
- 動画特徴 = フレーム特徴の「平均」「標準偏差」「フレーム間差分の平均」を連結
- コサイン類似度でランキング

## トラブルシュート
- モデルDL不可: ネット制限がある場合は `yolov8n-pose.pt` を手動DLして `--model` でパス指定
- 動画が読めない: コーデック依存。`ffmpeg`等で mp4(H.264) に変換
- 遅い: `--frame-stride` を上げる、`--max-frames` を下げる、可能ならGPUを使用

## ライセンス
データやモデルの利用規約は各自ご確認ください。ソースコードは本リポジトリに含まれるライセンスに準じます。

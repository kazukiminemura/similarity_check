# Video Similarity by Pose (YOLOv8‑Pose, OpenVINO)

Find videos whose pose is most similar to a target video. The app uses Ultralytics YOLOv8‑Pose to extract keypoints, builds a per‑video descriptor, and ranks candidates by cosine similarity.

This project runs inference with OpenVINO only (no PyTorch/CUDA fallback). Device selection (GPU/CPU/AUTO) is exposed in the web UI and mapped to OpenVINO plugins.

## Requirements

- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: ultralytics, openvino, opencv‑python, numpy, scikit‑learn, tqdm, fastapi, uvicorn, jinja2.

## Run the Web API

```bash
python -m similarity_check.web_api
# then open http://127.0.0.1:8000
```

### Video Roots

- Target videos default to `<repo>/target`
- Reference videos default to `<repo>/pro_data`

Override via environment variables (relative paths resolve from the repo root):

```bash
# PowerShell
$env:TARGET_ROOT = "C:\path\to\target"
$env:REFERENCE_ROOT = "C:\path\to\reference"
python -m similarity_check.web_api
```

The server mounts static video routes:
- `/videos/target` → TARGET_ROOT
- `/videos/reference` → REFERENCE_ROOT

## Web UI

Controls in the header:
- Target video: dropdown populated from TARGET_ROOT
- OpenVINO Device: GPU (Intel) / CPU / AUTO (default)
- TopK: number of results to show
- Frame stride: sample every N‑th frame
- Swing only: focus evaluation on a short, motion‑peak window
- Window (s): swing window length in seconds

Results render the target and top‑K similar candidates, playable and time‑syncable.

## API

- GET `/api/videos`
  - Returns: `{ target_root, reference_root, videos: [..] }`
- POST `/api/search`
  - Body (JSON):
    - `target`: string (file name under TARGET_ROOT or absolute path)
    - `device`: one of `gpu`, `cpu`, `auto` (mapped to OpenVINO `GPU`/`CPU`/`AUTO`)
    - `topk`: int (default 5)
    - `frame_stride`: int (default 5)
    - `swing_only`: bool (default true)
    - `swing_seconds`: number (default 2.5)
  - Response: `{ used_device, target: {path,name,url}, results: [{path,name,score,url}, ...] }`

### Feature Caching

Descriptors are cached as `.npz` in:
- `features_cache/` for full‑video
- `features_cache_swing/` for swing‑only

## How It Works

For each sampled frame, the app selects the most confident person (17 keypoints), normalizes for translation/scale, and aggregates over time using mean, std, and mean absolute frame‑to‑frame motion to form the video vector. Similarity is cosine distance.

Swing‑only mode detects a motion peak and centers a fixed‑length window (default ~2.5s, adjusted by FPS and stride).

## Troubleshooting

- “Invalid CUDA device …” when selecting GPU:
  - The backend uses OpenVINO. Some environments may misinterpret a device string; the server defensively retries without a device argument or falls back to CPU for the request. Prefer `AUTO` if unsure.
- OpenVINO export missing:
  - On first run, Ultralytics exports `yolov8n-pose.pt` to an OpenVINO model directory next to the weights. Ensure `openvino` is installed and restart if needed.

## License

See the repository license for source code terms. Ensure you have rights to any video content you process.
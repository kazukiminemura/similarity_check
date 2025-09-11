@echo off
setlocal

REM Usage: run_cli.bat <target_video> <candidates_dir_or_video> [topk] [frame_stride]

if "%~1"=="" (
  echo Usage: run_cli.bat ^<target_video^> ^<candidates^> [topk] [frame_stride]
  echo Example: run_cli.bat data\front_target1.mp4 data 5 5
  goto :eof
)

set TARGET=%~1
set CANDS=%~2
set TOPK=%~3
if "%TOPK%"=="" set TOPK=5
set STRIDE=%~4
if "%STRIDE%"=="" set STRIDE=5

if not exist .venv (
  echo [*] Creating virtual environment .venv
  py -m venv .venv
)

call .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt

python -m similarity_check.cli --target "%TARGET%" --candidates "%CANDS%" --topk %TOPK% --frame-stride %STRIDE% --model yolov8n-pose.pt

endlocal

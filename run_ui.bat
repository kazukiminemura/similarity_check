@echo off
setlocal

REM Create venv if missing
if not exist .venv (
  echo [*] Creating virtual environment .venv
  py -m venv .venv
)

call .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
pip install -r requirements-ui.txt

echo [*] Launching Streamlit UI
streamlit run similarity_check\app_streamlit.py

endlocal

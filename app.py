"""  """"""
Expose FastAPI application as `app` for `uvicorn app:app`.

Usage:
  uvicorn app:app --host 127.0.0.1 --port 8000 --reload
"""

# Simply re-export the FastAPI instance from the package module.
from similarity_check.web_api import app as app  # noqa: F401


"""
Web service for image-based video search.
- POST /api/search: upload image (multipart), run retrieval, return results. Uploaded file is not persisted; temp file is deleted after search.
- GET /frames/<path>: serve indexed frame images from data/frames.
- GET /: static search page.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# macOS: allow multiple OpenMP runtimes (PyTorch, faiss, etc.) to coexist.
if "KMP_DUPLICATE_LIB_OK" not in os.environ:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"

from flask import Flask, jsonify, request, send_from_directory

from . import config
from .search import search_by_image

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload


def _frame_path_to_url(frame_path: str) -> str:
    """Convert absolute frame_path from metadata to URL path under /frames/."""
    frames_dir = config.paths.frames_dir
    try:
        rel = os.path.relpath(frame_path, frames_dir)
    except ValueError:
        # different drives on Windows
        return ""
    rel = rel.replace("\\", "/")
    if rel.startswith("..") or os.path.isabs(rel):
        return ""
    return f"/frames/{rel}"


@app.route("/")
def index():
    """Serve the search page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return send_from_directory(STATIC_DIR, "index.html")
    return "Static page not found. Create static/index.html.", 404


@app.route("/frames/<path:subpath>")
def serve_frame(subpath: str):
    """Serve frame images from data/frames. No directory traversal."""
    if ".." in subpath or subpath.startswith("/"):
        return "Forbidden", 403
    frames_dir = config.paths.frames_dir
    return send_from_directory(frames_dir, subpath)


@app.route("/api/search", methods=["POST"])
def api_search():
    """
    Accept multipart image, run search, return JSON results.
    Uploaded file is written to a temp file only for the duration of the request and is deleted before returning.
    """
    if "image" not in request.files and "file" not in request.files:
        return jsonify({"error": "No image file (use field 'image' or 'file')"}), 400
    file = request.files.get("image") or request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    allowed = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if file.content_type and file.content_type not in allowed:
        return jsonify({"error": "Unsupported type; use JPEG/PNG/WebP/GIF"}), 400

    top_k = request.args.get("top_k", type=int, default=10)
    top_k = min(max(1, top_k), 50)
    index_dir = request.args.get("index_dir") or config.paths.index_dir

    suffix = Path(file.filename).suffix or ".jpg"
    if suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        suffix = ".jpg"
    fd = None
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="imgsearch_")
        file.save(temp_path)
        os.close(fd)
        fd = None
        results = search_by_image(
            image_path=temp_path,
            top_k=top_k,
            index_dir=index_dir,
        )
        out = [
            {
                "score": r.score,
                "video_path": r.metadata.get("video_path"),
                "video_id": r.metadata.get("video_id"),
                "timestamp_sec": r.metadata.get("timestamp_sec"),
                "frame_index": r.metadata.get("frame_index"),
                "frame_url": _frame_path_to_url(r.metadata.get("frame_path") or ""),
            }
            for r in results
        ]
        return jsonify({"results": out})
    except FileNotFoundError as e:
        return jsonify({"error": f"Index not found: {e}"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def main(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    if not STATIC_DIR.exists():
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()

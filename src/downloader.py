"""Download videos from a URL list via yt-dlp, extract keyframes, and optionally build index.

Videos are downloaded to a temporary path, keyframes are extracted, then the
video file is deleted so only frames and index need to be stored.
Requires yt-dlp to be installed on the system (e.g. pip install yt-dlp or system package).
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from typing import List, Optional
from urllib.parse import urlparse

from tqdm import tqdm

from .config import SceneDetectConfig, ensure_directories, paths
from .video_extractor import (
    FrameMetadata,
    extract_keyframes_from_video,
    _ensure_parent_dir,
)


def read_url_list(path: str) -> List[str]:
    """Read URLs from a text file (one URL per line; blank lines and # comments ignored)."""
    urls: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def _video_id_from_url(url: str, index: int) -> str:
    """Produce a stable, filesystem-safe video_id from URL and index."""
    # Use path or netloc for readability; fallback to hash.
    parsed = urlparse(url)
    name = (parsed.path or "/").rstrip("/").split("/")[-1] or parsed.netloc or url
    # Strip extension and sanitize.
    name = re.sub(r"\.[a-zA-Z0-9]+$", "", name)
    name = re.sub(r"[^\w\-.]", "_", name)[:80]
    digest = hashlib.md5(url.encode()).hexdigest()[:10]
    return f"url_{index:04d}_{digest}_{name}" if name else f"url_{index:04d}_{digest}"


def _find_downloaded_video(dir_path: str) -> Optional[str]:
    """Return path to the main video file in dir_path (skip .part and metadata)."""
    video_exts = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv")
    candidates = []
    for name in os.listdir(dir_path):
        if name.endswith(".part") or name.endswith(".ytdl"):
            continue
        if any(name.lower().endswith(ext) for ext in video_exts):
            candidates.append(os.path.join(dir_path, name))
    if not candidates:
        return None
    # Prefer mp4, then by size (largest is usually the full video).
    candidates.sort(key=lambda p: (not p.lower().endswith(".mp4"), -os.path.getsize(p)))
    return candidates[0]


def download_video_ytdlp(url: str, output_path: str) -> Optional[str]:
    """Download a video using yt-dlp to output_path (directory); returns path to video file.

    output_path should be a directory. yt-dlp will create a file like video.%(ext)s inside.
    Returns the path to the downloaded video file, or None on failure.
    """
    os.makedirs(output_path, exist_ok=True)
    out_tpl = os.path.join(output_path, "video.%(ext)s")
    try:
        subprocess.run(
            [
                "yt-dlp",
                "--no-playlist",
                "--no-warnings",
                "-o",
                out_tpl,
                url,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        if isinstance(e, FileNotFoundError):
            raise RuntimeError(
                "yt-dlp not found. Install it (e.g. pip install yt-dlp or system package) and ensure it is on PATH."
            ) from e
        raise
    return _find_downloaded_video(output_path)


def download_and_extract_from_url_list(
    url_list_path: str,
    frames_dir: Optional[str] = None,
    metadata_path: Optional[str] = None,
    temp_dir: Optional[str] = None,
    scene_cfg: Optional[SceneDetectConfig] = None,
) -> List[FrameMetadata]:
    """For each URL in the file: download to temp, extract keyframes, delete video.

    Frames are written under frames_dir; metadata is appended to metadata_path.
    video_path in frame metadata is set to the source URL (no local video kept).
    """
    ensure_directories()
    frames_dir = frames_dir or paths.frames_dir
    metadata_path = metadata_path or paths.frames_metadata_path
    data_dir = getattr(paths, "data_dir", None) or os.path.dirname(paths.frames_dir)
    temp_dir = temp_dir or os.path.join(data_dir, "tmp_downloads")
    scene_cfg = scene_cfg or SceneDetectConfig()

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    _ensure_parent_dir(metadata_path)

    urls = read_url_list(url_list_path)
    if not urls:
        raise ValueError(f"No URLs found in {url_list_path}")

    all_metas: List[FrameMetadata] = []
    for i, url in enumerate(tqdm(urls, desc="Videos", unit="video")):
        video_id = _video_id_from_url(url, i)
        temp_video_dir = tempfile.mkdtemp(prefix=video_id + "_", dir=temp_dir)
        try:
            video_path = download_video_ytdlp(url, temp_video_dir)
            if not video_path or not os.path.isfile(video_path):
                tqdm.write(f"Skip (no video file): {url}")
                continue
            # Extract keyframes; override video_path in metadata to store URL.
            metas = extract_keyframes_from_video(
                video_path,
                frames_dir,
                scene_cfg,
                video_id_override=video_id,
                video_display_path=url,
            )
            all_metas.extend(metas)
        finally:
            if os.path.isdir(temp_video_dir):
                try:
                    shutil.rmtree(temp_video_dir)
                except OSError:
                    pass

    # Write full JSONL metadata.
    with open(metadata_path, "w", encoding="utf-8") as f:
        for meta in all_metas:
            f.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")

    return all_metas

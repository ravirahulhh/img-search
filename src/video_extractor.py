import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional

import cv2
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from tqdm import tqdm

from .config import SceneDetectConfig, ensure_directories, paths


@dataclass
class FrameMetadata:
    video_path: str
    video_id: str
    frame_index: int
    timestamp_sec: float
    frame_path: str


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def detect_scenes(video_path: str, scene_cfg: SceneDetectConfig) -> List:
    """Run PySceneDetect content-based scene detection on a single video."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=scene_cfg.threshold,
            min_scene_len=scene_cfg.min_scene_len,
        )
    )
    scene_manager.detect_scenes(video)
    return scene_manager.get_scene_list()


def extract_keyframes_from_video(
    video_path: str,
    output_dir: str,
    scene_cfg: SceneDetectConfig,
    *,
    video_id_override: Optional[str] = None,
    video_display_path: Optional[str] = None,
) -> List[FrameMetadata]:
    """Extract representative keyframes from scenes of a single video.

    If video_display_path is set (e.g. source URL), it is stored in metadata
    as video_path instead of the local file path. Use with video_id_override
    when processing downloaded URLs so frame dirs and metadata are consistent.
    """
    ensure_directories()

    video_id = video_id_override or os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_output_dir, exist_ok=True)

    scenes = detect_scenes(video_path, scene_cfg)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_metas: List[FrameMetadata] = []

    for scene_idx, (start_time, end_time) in enumerate(scenes):
        # PySceneDetect returns timecodes; convert to frame indices.
        start_frame = start_time.get_frames()
        end_frame = end_time.get_frames()
        if end_frame <= start_frame:
            continue

        # Choose up to max_frames_per_scene uniformly within the scene.
        frame_indices: List[int] = []
        length = end_frame - start_frame
        max_frames = max(1, scene_cfg.max_frames_per_scene)
        step = max(1, length // (max_frames + 1))
        for i in range(1, max_frames + 1):
            idx = start_frame + i * step
            if idx >= end_frame:
                break
            frame_indices.append(idx)

        for local_idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            timestamp_sec = frame_idx / fps
            frame_filename = f"scene{scene_idx:05d}_k{local_idx:02d}.jpg"
            frame_path = os.path.join(video_output_dir, frame_filename)
            _ensure_parent_dir(frame_path)
            cv2.imwrite(frame_path, frame)
            display_path = video_display_path if video_display_path else os.path.abspath(video_path)
            frame_metas.append(
                FrameMetadata(
                    video_path=display_path,
                    video_id=video_id,
                    frame_index=int(frame_idx),
                    timestamp_sec=float(timestamp_sec),
                    frame_path=os.path.abspath(frame_path),
                )
            )

    cap.release()
    return frame_metas


def iter_videos(video_dir: str) -> Iterable[str]:
    for root, _, files in os.walk(video_dir):
        for name in files:
            if name.lower().endswith((".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv")):
                yield os.path.join(root, name)


def extract_keyframes_for_directory(
    video_dir: Optional[str] = None,
    frames_dir: Optional[str] = None,
    metadata_path: Optional[str] = None,
    scene_cfg: Optional[SceneDetectConfig] = None,
) -> List[FrameMetadata]:
    """Extract keyframes for all videos in a directory and write JSONL metadata."""
    ensure_directories()

    video_dir = video_dir or paths.videos_dir
    frames_dir = frames_dir or paths.frames_dir
    metadata_path = metadata_path or paths.frames_metadata_path
    scene_cfg = scene_cfg or SceneDetectConfig()

    os.makedirs(frames_dir, exist_ok=True)
    _ensure_parent_dir(metadata_path)

    all_metas: List[FrameMetadata] = []
    video_paths = list(iter_videos(video_dir))

    with tqdm(total=len(video_paths), desc="Extracting keyframes", unit="video") as pbar:
        for vp in video_paths:
            metas = extract_keyframes_from_video(vp, frames_dir, scene_cfg)
            all_metas.extend(metas)
            pbar.update(1)

    # Write JSONL metadata (one FrameMetadata per line).
    with open(metadata_path, "w", encoding="utf-8") as f:
        for meta in all_metas:
            f.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")

    return all_metas


import os
from dataclasses import dataclass


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PathConfig:
    data_dir: str = os.path.join(BASE_DIR, "data")
    videos_dir: str = os.path.join(data_dir, "videos")
    frames_dir: str = os.path.join(data_dir, "frames")
    index_dir: str = os.path.join(data_dir, "index")
    frames_metadata_path: str = os.path.join(data_dir, "frames_meta.jsonl")


@dataclass
class SceneDetectConfig:
    detector: str = "content"  # currently only content detector is used
    threshold: float = 27.0
    min_scene_len: int = 15  # in frames (depends on fps)
    max_frames_per_scene: int = 1  # limit frames per scene to control index size


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"
    batch_size: int = 32


@dataclass
class IndexConfig:
    nlist: int = 4096
    m: int = 64
    nbits: int = 8
    nprobe: int = 16


paths = PathConfig()
scene = SceneDetectConfig()
model = ModelConfig()
index = IndexConfig()


def ensure_directories() -> None:
    """Ensure that expected data directories exist."""
    os.makedirs(paths.videos_dir, exist_ok=True)
    os.makedirs(paths.frames_dir, exist_ok=True)
    os.makedirs(paths.index_dir, exist_ok=True)


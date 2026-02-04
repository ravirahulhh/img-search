import argparse
import os
from typing import Optional

# macOS: allow multiple OpenMP runtimes (PyTorch, faiss, etc.) to coexist.
if "KMP_DUPLICATE_LIB_OK" not in os.environ:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Reduce OpenMP/BLAS threads to avoid segfaults on macOS.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"

from . import config
from .downloader import download_and_extract_from_url_list
from .indexer import build_pq_index
from .search import search_by_image
from .video_extractor import extract_keyframes_for_directory


def cmd_extract_frames(args: argparse.Namespace) -> None:
    video_dir = args.video_dir or config.paths.videos_dir
    frames_dir = args.frames_dir or config.paths.frames_dir
    metadata_path = args.metadata_path or config.paths.frames_metadata_path

    print(f"Video dir: {video_dir}")
    print(f"Frames dir: {frames_dir}")
    print(f"Metadata: {metadata_path}")

    extract_keyframes_for_directory(
        video_dir=video_dir,
        frames_dir=frames_dir,
        metadata_path=metadata_path,
        scene_cfg=config.scene,
    )
    print("Keyframe extraction finished.")


def cmd_build_index(args: argparse.Namespace) -> None:
    frames_metadata = args.frames_metadata or config.paths.frames_metadata_path
    index_dir = args.index_dir or config.paths.index_dir

    print(f"Frames metadata: {frames_metadata}")
    print(f"Index dir: {index_dir}")

    paths_obj = build_pq_index(
        frames_metadata_path=frames_metadata,
        index_dir=index_dir,
        cfg=config.index,
    )
    print(f"Index saved to: {paths_obj.index_path}")
    print(f"Metadata saved to: {paths_obj.metadata_path}")


def cmd_index_from_urls(args: argparse.Namespace) -> None:
    url_list_path = args.url_list
    frames_dir = args.frames_dir or config.paths.frames_dir
    metadata_path = args.metadata_path or config.paths.frames_metadata_path
    index_dir = args.index_dir or config.paths.index_dir
    temp_dir = args.temp_dir

    print(f"URL list: {url_list_path}")
    print(f"Frames dir: {frames_dir}")
    print(f"Metadata: {metadata_path}")
    print("Downloading each video, extracting keyframes, then deleting the video file.")

    metas = download_and_extract_from_url_list(
        url_list_path=url_list_path,
        frames_dir=frames_dir,
        metadata_path=metadata_path,
        temp_dir=temp_dir,
        scene_cfg=config.scene,
    )
    print(f"Extracted {len(metas)} keyframes from {url_list_path}.")

    if not args.no_build_index:
        print("Building FAISS index...")
        paths_obj = build_pq_index(
            frames_metadata_path=metadata_path,
            index_dir=index_dir,
            cfg=config.index,
        )
        print(f"Index saved to: {paths_obj.index_path}")
        print(f"Metadata saved to: {paths_obj.metadata_path}")
    else:
        print("Skipping index build (--no-build-index). Run build-index to index the frames.")


def cmd_search(args: argparse.Namespace) -> None:
    image_path = args.image
    index_dir = args.index_dir or config.paths.index_dir
    top_k = args.top_k

    results = search_by_image(
        image_path=image_path,
        top_k=top_k,
        index_dir=index_dir,
    )
    if not results:
        print("No results found.")
        return

    print(f"Top {len(results)} results:")
    for i, r in enumerate(results, start=1):
        meta = r.metadata
        print(
            f"{i:02d}. score={r.score:.4f}, "
            f"video={meta.get('video_path')}, "
            f"timestamp={meta.get('timestamp_sec'):.2f}s, "
            f"frame={meta.get('frame_path')}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image-based video search PoC CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # extract-frames
    p_extract = subparsers.add_parser(
        "extract-frames", help="Extract keyframes from videos using PySceneDetect"
    )
    p_extract.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory containing input videos (default: config.paths.videos_dir)",
    )
    p_extract.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Directory to store extracted frames (default: config.paths.frames_dir)",
    )
    p_extract.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Path to JSONL metadata file (default: config.paths.frames_metadata_path)",
    )
    p_extract.set_defaults(func=cmd_extract_frames)

    # index-from-urls
    p_urls = subparsers.add_parser(
        "index-from-urls",
        help="Read URLs from file; download each video, extract keyframes (no video kept), then build index",
    )
    p_urls.add_argument(
        "--url-list",
        type=str,
        required=True,
        help="Path to text file with one video URL per line (blank lines and # comments ignored)",
    )
    p_urls.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Directory to store extracted frames (default: config.paths.frames_dir)",
    )
    p_urls.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Path to JSONL metadata file (default: config.paths.frames_metadata_path)",
    )
    p_urls.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Directory to store FAISS index (default: config.paths.index_dir)",
    )
    p_urls.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Temporary directory for downloads (default: data/tmp_downloads); deleted after each video",
    )
    p_urls.add_argument(
        "--no-build-index",
        action="store_true",
        help="Only download and extract; do not build the FAISS index",
    )
    p_urls.set_defaults(func=cmd_index_from_urls)

    # build-index
    p_index = subparsers.add_parser(
        "build-index", help="Build FAISS PQ index from frame embeddings"
    )
    p_index.add_argument(
        "--frames-metadata",
        type=str,
        default=None,
        help="Path to frames metadata JSONL file (default: config.paths.frames_metadata_path)",
    )
    p_index.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Directory to store FAISS index and index metadata (default: config.paths.index_dir)",
    )
    p_index.set_defaults(func=cmd_build_index)

    # search
    p_search = subparsers.add_parser(
        "search", help="Search similar frames given a query image"
    )
    p_search.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to query image file",
    )
    p_search.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Directory containing FAISS index (default: config.paths.index_dir)",
    )
    p_search.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )
    p_search.set_defaults(func=cmd_search)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()


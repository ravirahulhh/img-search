import argparse

from src.config import paths
from src.indexer import build_pq_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS index from extracted frame metadata."
    )
    parser.add_argument(
        "--frames-meta",
        type=str,
        default=paths.frames_metadata_path,
        help=f"Path to frames_meta.jsonl (default: {paths.frames_metadata_path})",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=paths.index_dir,
        help=f"Directory to store index files (default: {paths.index_dir})",
    )

    args = parser.parse_args()

    paths_obj = build_pq_index(
        frames_metadata_path=args.frames_meta,
        index_dir=args.index_dir,
    )

    print("Index built successfully.")
    print(f"FAISS index: {paths_obj.index_path}")
    print(f"Metadata:    {paths_obj.metadata_path}")


if __name__ == "__main__":
    main()


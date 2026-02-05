import argparse

from src.config import paths, index as index_cfg
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
    parser.add_argument(
        "--no-ivfpq",
        action="store_true",
        help="Disable IVFPQ and always use exact IndexFlatL2 when building the FAISS index",
    )

    args = parser.parse_args()

    # Apply CLI override for IVFPQ usage.
    allow_ivfpq = not args.no_ivfpq
    index_cfg.allow_ivfpq = allow_ivfpq

    paths_obj = build_pq_index(
        frames_metadata_path=args.frames_meta,
        index_dir=args.index_dir,
        cfg=index_cfg,
        allow_ivfpq=allow_ivfpq,
    )

    print("Index built successfully.")
    print(f"FAISS index: {paths_obj.index_path}")
    print(f"Metadata:    {paths_obj.metadata_path}")


if __name__ == "__main__":
    main()


"""
Image-based video search PoC package.

Modules:
- config: global configuration for paths and parameters.
- video_extractor: keyframe extraction via PySceneDetect.
- embedding: OpenCLIP-based image embedding.
- indexer: FAISS PQ/IVF index building and persistence.
- search: index loading and similarity search utilities.
- cli: command-line interface entrypoint.
"""


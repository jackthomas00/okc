"""
Stage 0: Ingestion & chunking.

This module handles:
- Document ingestion and normalization
- Text chunking with overlap
- Document filtering
"""

from okc_pipeline.stage_00_ingestion.chunker import chunk_text

__all__ = ["chunk_text"]


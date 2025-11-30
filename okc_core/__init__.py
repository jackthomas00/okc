"""
OKC Core - Shared types, config, database models, and schemas.

This package contains:
- Database models (SQLAlchemy)
- Database connection and session management
- Database migrations
- Configuration settings
- Pydantic schemas for API
"""

from okc_core.config import settings
from okc_core.db import Base, engine, SessionLocal
from okc_core.models import Document, Chunk, Entity, Sentence, EntityMention, ClaimSentence
from okc_core.schemas import (
    IngestRequest,
    BulkIngestRequest,
    IngestResult,
    SearchResponseItem,
    EntitySearchResult,
    UnifiedSearchResult,
)

__all__ = [
    "settings",
    "Base",
    "engine",
    "SessionLocal",
    "Document",
    "Chunk",
    "Entity",
    "Sentence",
    "EntityMention",
    "ClaimSentence",
    "IngestRequest",
    "BulkIngestRequest",
    "IngestResult",
    "SearchResponseItem",
    "EntitySearchResult",
    "UnifiedSearchResult",
]

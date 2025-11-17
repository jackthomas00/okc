from sqlalchemy import (
    Column, Integer, String, Text, ForeignKey, DateTime, UniqueConstraint, Index, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from api.db import Base

class Document(Base):
    __tablename__ = "document"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_url: Mapped[str | None] = mapped_column(Text)
    source_type: Mapped[str | None] = mapped_column(String(32))
    title: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[DateTime | None] = mapped_column(DateTime)

    text: Mapped[str] = mapped_column(Text)                   # original content
    lang: Mapped[str | None] = mapped_column(String(8))
    word_count: Mapped[int | None] = mapped_column(Integer)   # quick filter/QA

    content_hash: Mapped[str] = mapped_column(String(64), index=True)  # SHA1/64
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
    doc_embedding = Column(Vector(384), nullable=True)

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document")

    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_document_content_hash"),
        Index("ix_document_lang_wc", "lang", "word_count"),
        Index("ix_document_doc_embedding", "doc_embedding", postgresql_using="ivfflat"),
    )

class Chunk(Base):
    __tablename__ = "chunk"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer, index=True)
    text: Mapped[str] = mapped_column(Text)
    embedding = Column(Vector(384))   # match your embedding dimension

    document: Mapped[Document] = relationship(back_populates="chunks")
    sentences: Mapped[list["Sentence"]] = relationship(back_populates="chunk")

    __table_args__ = (
        Index("ix_chunk_embedding", "embedding", postgresql_using="ivfflat"),
    )

class Sentence(Base):
    __tablename__ = "sentence"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunk.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    order_index: Mapped[int] = mapped_column(Integer)

    chunk: Mapped[Chunk] = relationship(back_populates="sentences")
    mentions: Mapped[list["EntityMention"]] = relationship(back_populates="sentence")

    __table_args__ = (
        Index("idx_sentence_order_index", "chunk_id", "order_index"),
    )

class Entity(Base):
    __tablename__ = "entity"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_name: Mapped[str] = mapped_column(Text)
    type: Mapped[str | None] = mapped_column(String(64))
    normalized_name: Mapped[str | None] = mapped_column(Text)
    extra_metadata = Column(JSONB, nullable=True)

    mentions: Mapped[list["EntityMention"]] = relationship(back_populates="entity")

    __table_args__ = (
        UniqueConstraint("canonical_name", name="uq_entity_canonical_name"),
        Index("idx_entity_normalized_name", "normalized_name"),
        Index("ix_entity_canonical_name_trgm", "canonical_name", postgresql_using="gin", postgresql_ops={"canonical_name": "gin_trgm_ops"}),
    )

class EntityMention(Base):
    __tablename__ = "entity_mention"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("entity.id", ondelete="CASCADE"), index=True)
    sentence_id: Mapped[int] = mapped_column(ForeignKey("sentence.id", ondelete="CASCADE"), index=True)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    surface_text: Mapped[str] = mapped_column(Text)

    entity: Mapped[Entity] = relationship(back_populates="mentions")
    sentence: Mapped[Sentence] = relationship(back_populates="mentions")

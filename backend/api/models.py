from sqlalchemy import (
    Column, Integer, String, Text, ForeignKey, Float, Enum, DateTime, UniqueConstraint, Index, func
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from api.db import Base

PolarityEnum = Enum("supports","contradicts","neutral", name="polarity_enum")
RelationEnum = Enum("is_a","part_of","influences","similar_to","contradicts","derived_from","uses","depends_on","improves", name="relation_enum")

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
    entities: Mapped[list["EntityChunk"]] = relationship(back_populates="chunk")

    __table_args__ = (
        Index("ix_chunk_embedding", "embedding", postgresql_using="ivfflat"),
    )

class Entity(Base):
    __tablename__ = "entity"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, index=True)
    type: Mapped[str | None] = mapped_column(String(64))
    description: Mapped[str | None] = mapped_column(Text)
    popularity: Mapped[float | None] = mapped_column(Float)
    canonical_label: Mapped[str | None] = mapped_column(Text)
    alias_of: Mapped[int | None] = mapped_column(ForeignKey("entity.id"))
    centroid = Column(Vector(384))

    __table_args__ = (
        UniqueConstraint("name", name="uq_entity_name"),
        Index("ix_entity_name_trgm", "name", postgresql_using="gin", postgresql_ops={"name": "gin_trgm_ops"}),
    )

class EntityChunk(Base):
    __tablename__ = "entity_chunk"
    entity_id: Mapped[int] = mapped_column(ForeignKey("entity.id", ondelete="CASCADE"), primary_key=True)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunk.id", ondelete="CASCADE"), primary_key=True)
    span_start: Mapped[int | None] = mapped_column(Integer)
    span_end: Mapped[int | None] = mapped_column(Integer)

    entity: Mapped[Entity] = relationship()
    chunk: Mapped[Chunk] = relationship(back_populates="entities")

class Topic(Base):
    __tablename__ = "topic"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    centroid = Column(Vector(384))

class TopicMember(Base):
    __tablename__ = "topic_member"
    topic_id: Mapped[int] = mapped_column(ForeignKey("topic.id", ondelete="CASCADE"), primary_key=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("entity.id", ondelete="CASCADE"), primary_key=True)
    score: Mapped[float] = mapped_column(Float)

class Claim(Base):
    __tablename__ = "claim"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    polarity = Column(PolarityEnum, nullable=False, default="neutral")
    confidence: Mapped[float] = mapped_column(Float, default=0.5)

class ClaimSource(Base):
    __tablename__ = "claim_source"
    claim_id: Mapped[int] = mapped_column(ForeignKey("claim.id", ondelete="CASCADE"), primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id", ondelete="CASCADE"), primary_key=True)
    chunk_id: Mapped[int | None] = mapped_column(ForeignKey("chunk.id", ondelete="CASCADE"), primary_key=True)

class Relation(Base):
    __tablename__ = "relation"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    head_entity_id: Mapped[int] = mapped_column(ForeignKey("entity.id", ondelete="CASCADE"), index=True)
    tail_entity_id: Mapped[int] = mapped_column(ForeignKey("entity.id", ondelete="CASCADE"), index=True)
    type = Column(RelationEnum, nullable=False)
    evidence_claim_id: Mapped[int | None] = mapped_column(ForeignKey("claim.id", ondelete="SET NULL"))
    confidence: Mapped[float] = mapped_column(Float, default=0.5)

    __table_args__ = (
        UniqueConstraint("head_entity_id","tail_entity_id","type", name="uq_relation_triplet"),
    )

from pydantic import BaseModel

class IngestRequest(BaseModel):
    title: str
    url: str | None = None
    text: str
    lang: str | None = None

class BulkIngestRequest(BaseModel):
    items: list[IngestRequest]

class IngestResult(BaseModel):
    document_id: int | None
    deduped: bool
    num_chunks: int
    title: str | None = None
    url: str | None = None

class SearchResponseItem(BaseModel):
    chunk_id: int
    document_id: int
    title: str | None
    snippet: str
    score: float

class EntitySearchResult(BaseModel):
    id: int
    name: str
    canonical_label: str | None
    score: float | None = None

class UnifiedSearchResult(BaseModel):
    id: int
    type: str  # "topic" | "entity" | "document"
    title: str
    snippet: str | None = None
    score: float

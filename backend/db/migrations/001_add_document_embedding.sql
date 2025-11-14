ALTER TABLE document
    ADD COLUMN IF NOT EXISTS doc_embedding vector(384);

CREATE INDEX IF NOT EXISTS ix_document_doc_embedding
    ON document
    USING ivfflat (doc_embedding);

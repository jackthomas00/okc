-- Migration 003: Milestone 1 Schema Restructure
-- Drops old tables and creates new Sentence, Entity, EntityMention tables
-- Preserves Document and Chunk tables

-- Drop tables in dependency order (CASCADE handles foreign keys)
DROP TABLE IF EXISTS relation CASCADE;
DROP TABLE IF EXISTS claim_source CASCADE;
DROP TABLE IF EXISTS claim CASCADE;
DROP TABLE IF EXISTS topic_member CASCADE;
DROP TABLE IF EXISTS topic CASCADE;
DROP TABLE IF EXISTS entity_chunk CASCADE;
DROP TABLE IF EXISTS entity CASCADE;

-- Drop enum types if they exist
DROP TYPE IF EXISTS relation_enum CASCADE;
DROP TYPE IF EXISTS polarity_enum CASCADE;

-- Create new Sentence table
CREATE TABLE sentence (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunk(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    order_index INTEGER NOT NULL
);

CREATE INDEX idx_sentence_chunk_id ON sentence(chunk_id);
CREATE INDEX idx_sentence_order_index ON sentence(chunk_id, order_index);

-- Create new Entity table
CREATE TABLE entity (
    id SERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    type VARCHAR(64),
    normalized_name TEXT,
    extra_metadata JSONB
);

CREATE UNIQUE INDEX uq_entity_canonical_name ON entity(canonical_name);
CREATE INDEX idx_entity_normalized_name ON entity(normalized_name);
CREATE INDEX ix_entity_canonical_name_trgm ON entity USING gin(canonical_name gin_trgm_ops);

-- Create EntityMention table
CREATE TABLE entity_mention (
    id SERIAL PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES entity(id) ON DELETE CASCADE,
    sentence_id INTEGER NOT NULL REFERENCES sentence(id) ON DELETE CASCADE,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    surface_text TEXT NOT NULL
);

CREATE INDEX idx_entity_mention_entity_id ON entity_mention(entity_id);
CREATE INDEX idx_entity_mention_sentence_id ON entity_mention(sentence_id);


-- Spot-check Relations and their supporting Claims
-- This query validates that relations make sense by checking:
-- 1. Relations have valid head/tail entities
-- 2. Relations have supporting claims
-- 3. Claim text context is shown
-- 4. Confidence levels are displayed

SELECT
    r.id as relation_id,
    r.type as relation_type,
    r.confidence as relation_confidence,
    e1.id as head_entity_id,
    e1.name as head_entity_name,
    e2.id as tail_entity_id,
    e2.name as tail_entity_name,
    c.id as claim_id,
    c.text as claim_text,
    c.confidence as claim_confidence,
    c.polarity as claim_polarity,
    -- Show chunk context if available
    ch.id as chunk_id,
    LEFT(ch.text, 200) as chunk_preview,
    d.title as document_title
FROM relation r
LEFT JOIN entity e1 ON r.head_entity_id = e1.id
LEFT JOIN entity e2 ON r.tail_entity_id = e2.id
LEFT JOIN claim c ON r.evidence_claim_id = c.id
LEFT JOIN claim_source cs ON c.id = cs.claim_id
LEFT JOIN chunk ch ON cs.chunk_id = ch.id
LEFT JOIN document d ON cs.document_id = d.id
WHERE 
    -- Filter out relations without claims (potential data quality issues)
    r.evidence_claim_id IS NOT NULL
ORDER BY 
    -- Show higher confidence relations first, then by relation ID
    r.confidence DESC NULLS LAST,
    r.id
LIMIT 40;
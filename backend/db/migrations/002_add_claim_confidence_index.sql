-- Add index on Claim.confidence to speed up filtering queries
CREATE INDEX IF NOT EXISTS ix_claim_confidence
    ON claim (confidence);


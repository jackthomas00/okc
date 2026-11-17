# 🧠 Open Knowledge Compiler (OKC)

**OKC** is an open-source, domain-agnostic system that turns **unstructured text** into a **multi-layer knowledge graph** with:

* **Entities** (canonicalized concepts)
* **Sentences** (structured text units)
* **Entity Mentions** (entity occurrences in sentences)
* **Claims** (evidence-bearing sentences)
* **Relations** (typed edges derived from text, each linked to its evidence sentence)
* **Provenance** (source documents + chunks)

It is *not* just RAG.
It's a **knowledge compiler**: a pipeline that extracts structure, semantics, and relationships from raw text and makes them queryable, explainable, and navigable.

---

## 📦 Package Structure

OKC is organized into modular packages:

```
okc/
├── okc_core/        # Shared types, config, database models, schemas
├── okc_pipeline/    # Ingestion + extraction + graph building
├── okc_api/         # FastAPI REST API
├── okc_cli/         # CLI entrypoints
├── okc_ui/          # React explorer UI
├── examples/        # Runnable sample: stages 1–4 on a short text, no Postgres needed
├── tests/           # Unit tests per stage, end-to-end pipeline and API tests
└── docs/            # Architecture, how-to, diagrams
```

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/jackthomas00/okc
cd okc
pip install -e .
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

This starts:
* `okc_api` — FastAPI service on port 8000
* `okc_db` — Postgres + pgvector
* `okc_frontend` — React dev server on port 5173

### 3. Ingest some documents

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "title":"Transformers Overview",
    "url":"https://example.com",
    "text":"Transformers are neural network architectures that rely on self-attention..."
  }'
```

Or use the provided **Wikipedia crawler**:

```bash
python -m okc_cli.wiki_crawler_adv
```

---

## 🔍 Example Output

Running stages 1–4 on this five-sentence input ([`examples/sample.txt`](examples/sample.txt)):

> The Transformer is a neural network architecture introduced by researchers at Google. The Transformer depends on self-attention. Self-attention is an attention mechanism. BERT requires large amounts of pretraining data. BERT was evaluated on the GLUE benchmark.

produces the following output (abridged). Every relation points back to the sentence it came from:

```json
{
  "entities": ["BERT", "GLUE", "Google", "Transformer", "self-attention", "an attention mechanism", "..."],
  "claims": [
    "The Transformer depends on self-attention.",
    "Self-attention is an attention mechanism.",
    "BERT was evaluated on the GLUE benchmark.",
    "..."
  ],
  "relations": [
    {
      "head": "Transformer",
      "relation": "is_a",
      "tail": "a neural network architecture",
      "confidence": 1.0,
      "evidence": "The Transformer is a neural network architecture introduced by researchers at Google."
    },
    {
      "head": "Transformer",
      "relation": "depends_on",
      "tail": "self-attention",
      "confidence": 0.9,
      "evidence": "The Transformer depends on self-attention."
    },
    {
      "head": "self-attention",
      "relation": "is_a",
      "tail": "an attention mechanism",
      "confidence": 1.0,
      "evidence": "Self-attention is an attention mechanism."
    }
  ]
}
```

To reproduce it, run the script below. It uses a throwaway SQLite database, so you don't need Docker or Postgres:

```bash
python -m spacy download en_core_web_md
python examples/run_sample.py examples/sample.txt
```

---

## ⚙️ Pipeline Robustness

The batch pipeline ([`okc_pipeline/pipeline_runner.py`](okc_pipeline/pipeline_runner.py)) is configured by [`okc_pipeline/pipeline.yaml`](okc_pipeline/pipeline.yaml). It has been used to crawl and process 5,000+ Wikipedia documents.

* **Rate-limit-aware crawling:** backs off on HTTP 429/503, honors `Retry-After` and adds random delays between requests.
* **Content-hash deduplication:** hashes document text with SHA-1, so the same article is never processed twice.
* **Restartable stages:** `--skip <stage>` and `--skip-sourcing` re-run only the stages you need on data already in the database. `--clear-db` forces a full rebuild.

```bash
# Full run
python -m okc_pipeline.pipeline_runner okc_pipeline/pipeline.yaml

# Re-run only claims + relations on documents already ingested
python -m okc_pipeline.pipeline_runner okc_pipeline/pipeline.yaml \
  --skip-sourcing --skip chunk --skip embed --skip sentences --skip entities
```

---

## 📚 Documentation

See the `docs/` folder for:
* `Architecture.md` - System architecture overview
* `Pipeline.md` - Pipeline stages and processing
* `NewDBSketch.md` - Database schema design

---

## 🏗 Development

### Running locally (without Docker)

1. Set up PostgreSQL with pgvector extension
2. Set environment variables:
   ```bash
   export DATABASE_URL="postgresql+psycopg://user:pass@localhost:5432/okc"
   export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
   ```
3. Tables are created automatically when the API starts (via SQLAlchemy Base.metadata.create_all)
4. Start API:
   ```bash
   uvicorn okc_api.main:app --reload
   ```
5. Start UI:
   ```bash
   cd okc_ui && npm install && npm run dev
   ```

### Running tests

```bash
pip install -e ".[dev]" && python -m spacy download en_core_web_md
DATABASE_URL=sqlite:///./test.db pytest tests/
```

See [`tests/README.md`](tests/README.md) for per-stage and end-to-end test options.

---

## 📝 License

MIT License

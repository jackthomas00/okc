# 🧠 Open Knowledge Compiler (OKC)

**OKC** is an open-source, domain-agnostic system that turns **unstructured text** into a **multi-layer knowledge graph** with:

* **Entities** (canonicalized concepts)
* **Sentences** (structured text units)
* **Entity Mentions** (entity occurrences in sentences)
* **Claims** (evidence-bearing sentences) - Milestone 2
* **Relations** (typed edges derived from text) - Milestone 2
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
├── examples/        # Example scripts ("run OKC on mini-wiki", "on arXiv")
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

---

## 📝 License

MIT License

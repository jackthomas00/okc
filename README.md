# 🧠 Open Knowledge Compiler (OKC)

**OKC** is an open-source, domain-agnostic system that turns **unstructured text** into a **multi-layer knowledge graph** with:

* **Entities** (canonicalized concepts)
* **Topics** (clusters of related entities)
* **Claims** (evidence-bearing sentences)
* **Relations** (typed edges derived from text)
* **Provenance** (source documents + chunks)

It is *not* just RAG.
It's a **knowledge compiler**: a pipeline that extracts structure, semantics, and relationships from raw text and makes them queryable, explainable, and navigable.

---

## 🔍 Why this exists

LLMs are powerful — but they:

* hallucinate,
* forget,
* lack persistent structure,
* and cannot reason over relationships without help.

Traditional RAG systems rely on **flat vector search**.
Knowledge graphs encode **relationships**, but existing ones (Wikidata, DBpedia, corporate KGs) are:

* manually curated,
* incomplete,
* domain-specific,
* or completely closed.

**OKC builds structure automatically** from text.
It gives you a **transparent, evidence-backed knowledge substrate** for retrieval, reasoning, and agents.

---

## 🌐 High-Level Architecture

```
raw text
   ↓
chunking → embeddings → dedupe
   ↓
entity extraction → alias merging → canonical identifiers
   ↓
co-mentions → claims → relation typing
   ↓
topic clustering (entity-level)
   ↓
storage (Postgres + pgvector)
   ↓
GraphQL/REST API
   ↓
React Explorer (search + entity pages + graph view)
```

---

## ✨ Features

### 🧩 Multi-Layer Knowledge Graph

* **Entities** extracted from chunks
* Canonicalization (alias merging using trigram + embedding similarity)
* Entity-chunk alignment with character spans
* Topic clustering using HDBSCAN/k-means on entity embeddings
* Claims and typed relations with supporting sentences

### 🔎 Semantic Search

* Cosine vector search using `pgvector`
* Hybrid retrieval: chunk embeddings + entity lookup

### 🕸 Graph Explorer UI

* Entity search
* Entity pages (neighbors, summaries)
* Graph view (1-hop neighborhood)
* Evidence modal (supporting sentences with citations)

### 🧪 Sources

OKC supports any text source.
Included examples:

* Wikipedia (API crawler or dump ingestion)
* arXiv / OpenAlex abstracts
* Local text corpora
* Custom PDFs after text extraction

### 📦 API

* `/search` — semantic chunk search
* `/entities/search` — entity autocomplete
* `/entity/{id}` — entity details + neighbors
* `/graph/entity/{id}` — graph slice
* `/topic/{id}` — topic info
* `/ingest` + `/ingest_bulk` — pipeline ingestion

---

## 🚀 Quick Start

### 1. Clone and run with Docker Compose

```bash
git clone https://github.com/jackthomas00/okc
cd okc
docker compose up --build
```

This starts:

* `okc_api` — FastAPI service on port 8000
* `okc_db` — Postgres + pgvector
* optional: `okc_ui` (React dev server)

---

### 2. Ingest some documents

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
python wiki_crawler_adv.py
```

---

### 3. Launch the Explorer UI

If running locally:

```bash
cd ui
npm install
npm run dev
```

Navigate to:

```
http://localhost:5173
```

Search for “Transformer” and click around.

---

## 🏗 Pipeline Overview

The ingestion pipeline performs:

### 1. Chunking

* Sentence-aware chunk splitting
* Target token length (configurable)
* Overlap handling

### 2. Embedding

* Sentence-transformers / MiniLM by default
* Embeddings normalized (cosine distance)

### 3. Entity Extraction

* Noun phrase + pattern-based extraction
* Basic NER
* Entity-chunk span recording

### 4. Alias Merging

Consolidate:

* acronym variants
* casing differences
* lexical edits
* embedding-proximal labels

### 5. Claim Extraction

* Pattern-based sentence scanning
* Detects:

  * “X is a Y”
  * “X uses Y”
  * “X improves Y”
  * “X depends on Y”
* Stores each claim with source chunk + confidence

### 6. Relation Typing

* Lightweight rules → typed edges
* `is_a`, `part_of`, `uses`, `depends_on`, `improves`, `similar_to`, etc.
* Supports contradictions & polarity

### 7. Topic Graph

* Cluster entities via HDBSCAN or k-means
* Label topics from member entities & key n-grams
* Summaries generated via LLM (optional)

---

## 📊 Storage Layer

**Postgres 16** with `pgvector` and `pg_trgm`.

Tables include:

* `document`
* `chunk`
* `entity`
* `entity_chunk`
* `claim`
* `claim_source`
* `relation`
* `topic`
* `topic_member`

Vector indexes on:

* `chunk.embedding` (IVFFlat)
* `entity.centroid` (optional)

Trigram indexes for:

* `entity.name`

---

## 🖼 UI Overview

Built in **React + Vite** with **Cytoscape.js**.

### Pages:

#### **Search**

* query → entity list
* hybrid text+embedding rank (optional)

#### **Entity Page**

* canonical name
* neighbors (co-mentions or relations)
* entity summary (optional)
* interactive graph

#### **Graph View**

* one-hop neighbor graph
* edge evidence modal

#### **Topic Page**

* label + summary
* member entities
* mini-graph of topic neighborhood

---

## 🧬 Why OKC is Different

### Not just RAG

RAG retrieves **chunks**.
OKC retrieves **structure**: entities, topics, relations, claims.

### Not just a knowledge graph

Traditional KGs are static and hand-curated.
OKC is dynamic, automatic, and multi-layer.

### Not just embeddings

Vector search alone has no semantics.
OKC uses embeddings to **construct** semantics.

### Not proprietary

Google, Amazon, OpenAI have internal equivalents.
There's no public, open alternative.

**OKC fills that gap.**

---

## 📈 Roadmap

### **v0.1**

* Chunking, embeddings, entity extraction
* Co-mention graphs
* Basic UI

### **v0.2**

* Topic clustering
* Claims + simple relation extraction
* Evidence modal

### **v1.0**

* Evaluated canonicalization
* Multi-source ingest (Wikipedia, arXiv, OpenAlex, PDFs)
* Constraint checks (is_a cycles, relation consistency)
* GraphQL API

### **v2.0**

* Subgraph reasoning API
* LLM-augmented summaries
* Graph-based question answering
* Agent integration

---

## 🔖 License

MIT (or your choice).

---

## 💡 Contributing

Coming soon.
Issues and PRs welcome.

---

# What you’re building

A pipeline + graph that turns unstructured text into **entities, topics, relations, and claims**; stores them in a queryable graph with provenance; and serves a **navigable “map of knowledge”** with an API and UI. Think: scrape → chunk → embed → cluster → extract → link → serve.

---

# MVP scope (narrow slice, high leverage)

**Domain focus (start):** AI/ML methods (e.g., “transformers, diffusion, retrieval-augmented generation”).
**Sources (start):** Wikipedia dumps, arXiv abstracts (metadata only), top tutorials/blogs with permissive licenses.
**What the MVP must do:**

1. Ingest ~5–10k documents.
2. Produce **Topics** (clusters), **Entities** (canonical terms), **Relations** between entities, and **Claims** (sentence-level facts) with **Source** provenance.
3. Serve a **GraphQL API** + **React UI** to explore: Topic → Entities → Claims → Sources, with graph view and search.
4. Show **why** two nodes are linked (supporting sentences).

---

# Core data model (Postgres + pgvector; optional Neo4j later)

**Tables**

* `document(id, source_url, source_type, title, published_at, text, lang)`
* `chunk(id, document_id, idx, text, embedding vector(768))`
* `entity(id, name, type, description, popularity, canonical_label, alias_of nullable)`
* `topic(id, label, summary, centroid vector(768))`
* `claim(id, text, polarity enum('supports','contradicts','neutral'), confidence float)`
* `relation(id, head_entity_id, tail_entity_id, type enum('is_a','part_of','influences','similar_to','contradicts','derived_from'), evidence_claim_id)`
* `entity_chunk(entity_id, chunk_id, span_start, span_end)`
* `topic_member(topic_id, entity_id, score)`
* `claim_source(claim_id, document_id, chunk_id)`

**Indexes**

* IVF/IVFFlat on `chunk.embedding` (pgvector)
* trigram index on `entity.name`, `document.title`
* composite on `relation(head_entity_id, tail_entity_id, type)`

**Why Postgres first?** You’ll get transactional safety + full-text + vectors in one place. If/when graph queries get heavy, mirror to Neo4j.

---

# Pipeline (stages and checks)

1. **Fetch**

   * Pull documents (dumps/APIs).
   * Normalize metadata (URL, date, author, license).
   * Store in `document`.

2. **Chunk**

   * Split text into ~500–800 token chunks with overlap.
   * Write to `chunk`; compute **embeddings** (sentence-transformers style dim 384–1024).
   * Q/A: dedupe near-duplicates via cosine > 0.95.

3. **Entity extraction**

   * NER + term mining (noun chunks + TF-IDF + cased heuristics).
   * Canonicalize (lowercase, strip parens), build alias map (string similarity + embedding similarity).
   * Insert `entity`; create `entity_chunk` spans.
   * Link aliases using union-find on similarity graph with thresholds.

4. **Topic clustering**

   * Cluster **entities** by embedding (HDBSCAN or k-means with silhouette sweeps).
   * For each cluster, compute centroid; label via top n-grams + mutual information terms.
   * Insert `topic` + `topic_member`.

5. **Claim extraction**

   * From chunks, extract candidate **claim sentences** (contain entity pairs and verbs like “is, causes, improves, depends on, outperforms”).
   * Lightweight relation typing (pattern + dependency rules).
   * Insert `claim`; attach `claim_source`.

6. **Relation linking**

   * From claims that reference ≥2 entities, infer `relation(type)` with a heuristic classifier.
   * Attach `evidence_claim_id`.
   * Consolidate duplicates by `(head, type, tail)`; keep best evidence + confidence aggregation.

7. **Quality passes**

   * Drop relations with low confidence AND no corroborating claims.
   * Human-in-the-loop review queue for top-degree nodes and controversial edges.

---

# API surface (GraphQL + minimal REST)

**GraphQL (Strawberry/Ariadne)**

* `topic(id|query) { id label summary entities { id name score } relatedTopics { id label } }`
* `entity(id|name) { id name type description topics { id label } relations { type to { id name } evidence { claim { id text } source { url } } }`
* `search(q: String!): [SearchResult]` → unions of Topic/Entity/Document
* `claim(id) { id text polarity confidence sources { url title } entities { id name } }`

**REST (FastAPI)**

* `GET /search?q=...` (quick autocomplete)
* `GET /graph/entity/{id}` (JSON for Cytoscape)

Auth: None for read in MVP. Rate-limit and cache.

---

# UI (React + Vite + Cytoscape.js)

* **Global search bar** → results (topics/entities/documents).
* **Topic page:** label, summary, key entities, mini-graph.
* **Entity page:** definition, related entities (by relation type tabs), **Evidence** pane showing claims/sentences.
* **Graph view:** center on a node, filter edges by type/confidence, click edge → evidence modal.
* **Provenance:** always visible link to source chunk/sentence.

---

# Repo structure

```
okc/
  pipeline/
    fetch/
    chunk/
    embed/
    entities/
    topics/
    claims/
    relations/
    common/ (db.py, logging.py, config.py)
  api/
    graphql/
    rest/
  ui/
    src/
  db/
    migrations/
    seed/
  scripts/
    run_ingest.py
    eval_quality.py
  tests/
  docker/
  README.md
```

---

# 90-day execution plan (deliverables, not fluff)

**Days 1–10 — Skeleton & storage**

* Postgres+pgvector + FastAPI bootstrapped.
* Document/chunk tables + migrations.
* Embedding service (simple worker).
* CLI: `okc ingest <path_or_url>`; `okc embed`.

**Days 11–20 — Entities**

* NER+term mining pass; aliasing + canonicalization.
* `entity_chunk` spans; quick entity search.
* Smoke tests: precision@k for entity search.

**Days 21–30 — Topics**

* Clustering entities; labels via top n-grams.
* Topic pages in UI; list member entities.
* Basic search UX.

**Days 31–45 — Claims & Relations**

* Sentence picker + relation heuristics (pattern+deps).
* `claim` + `relation` tables wired with provenance.
* Graph view (Cytoscape) from entity → neighbors.

**Days 46–60 — Evidence UX + Quality**

* Edge click → show evidence sentences with sources.
* Confidence scoring (combine: pattern strength, redundancy across docs, source credibility).
* Review queue: JSON export of low-confidence/high-impact edges.

**Days 61–75 — GraphQL + perf**

* Full GraphQL schema + resolvers.
* Caching layer (Redis) for heavy queries.
* Index audits (vector, trigram, composite).

**Days 76–90 — Polish + release**

* Domain seed (5–10k docs).
* README with architecture + contribution guide.
* Demo dataset dump + Docker Compose for one-command run.
* v0.1 tag.

---

# Heuristics that matter (bluntly)

* **Aliases will kill you** if you don’t dedupe hard: use **(name trigram > 0.85) OR (embedding > 0.9)**, then human spot-check top merges.
* **Clustering brittle?** Start k-means with k grid (50–300), pick by elbow + silhouette; switch to HDBSCAN later.
* **Relations noisy?** Require **≥2 supporting claims** from **≥2 sources** for public display; show single-source edges behind a “low confidence” toggle.
* **Latency**: precompute 1-hop neighborhoods nightly; keep graph queries snappy.

---

# Minimal algorithms (simple, workable)

**Relation inference rule-of-thumb**

* Dependency patterns:

  * `X (nsubj) VERB (lemma in {improves, enables, uses, depends, causes, replaces}) Y (dobj/pobj)` → map verb to relation type.
* If multiple verbs appear with same (X,Y), pick most common; increase confidence with redundancy.

**Topic labeling**

* From top member entities + top chunk n-grams near centroid; pick 2–4 terms; no more.

---

# Evaluation (you need this)

* **Entity canonicalization accuracy:** sample 200 merges → target ≥90% correct.
* **Relation precision@20:** manual review of top-degree pairs → ≥80% acceptable.
* **Coverage:** % of chunks with ≥1 entity; % entities with ≥1 relation.
* **Latency:** p95 for `entity(id)->neighbors` ≤ 200ms on dev data.

---

# Tech stack picks (pragmatic)

* **Backend:** Python (FastAPI), SQLModel/SQLAlchemy.
* **Embeddings:** sentence-transformers-style local model (dimension 384–1024).
* **DB:** Postgres 16 + pgvector.
* **Queue:** simple Celery/RQ worker or just asyncio jobs to start.
* **GraphQL:** Strawberry or Ariadne.
* **UI:** React + Vite + TanStack Query + Cytoscape.js.

---

# Licenses & posture

* **Code:** MIT/Apache-2.0.
* **Data outputs:** CC-BY 4.0 (preserve attribution, enable reuse).
* **Provenance:** store `source_url` + paragraph offsets. Don’t redistribute full copyrighted texts—only short claim sentences with citation.

---

# Extensions after v0.1

* **Neo4j mirror** for advanced path queries.
* **Active learning loop:** review UI that writes feedback back into alias/edge models.
* **Temporal edges:** add `valid_from/valid_to`; show how knowledge changes.
* **Contradiction surfacing:** cluster claims that disagree; show sources side-by-side.
* **Agent hooks:** API to fetch **grounded subgraphs** for a query (great for RAG).

---

# Dedupe + doc embeddings

* Every document now stores a doc-level vector (mean of its chunk embeddings) so we can drop near-duplicates at ingest time via cosine > 0.95.
* Run `python scripts/dedupe_docs.py --threshold 0.95 --output duplicates.json` to backfill vectors for existing docs and emit a JSON report of suspicious pairs. Add `--skip-backfill` if you only need the report.
* The ingestion API computes a lightweight paragraph embedding before insert, queries the nearest neighbors, and rejects any document whose best match clears the threshold; once chunk embeddings are stored, we refresh the doc vector with the mean of the actual chunk embeddings.

# First tasks you can do today

1. Scaffold Postgres + pgvector + FastAPI and create the schema above.
2. Write `chunker.py` and `embedder.py` (CLI).
3. Build a toy ingest from a handful of public Wikipedia pages saved locally to prove: doc → chunks → embeddings → entity extraction → one topic page in React.

If you want, I’ll draft the **initial SQL schema + FastAPI models + a tiny entity extraction function** so you can paste it in and run `docker compose up` and see the bones working.

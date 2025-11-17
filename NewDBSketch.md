## 3. DB schema sketch

You can adapt to your existing tables, but here’s a minimal, sane schema:

### Core text

```sql
Document(id, source_uri, title, metadata_json)

Chunk(
  id,
  document_id REFERENCES Document,
  text,
  embedding VECTOR,
  order_index INT
)

Sentence(
  id,
  chunk_id REFERENCES Chunk,
  text,
  char_start INT,
  char_end INT,
  order_index INT
)
```

### Graph: entities & mentions

```sql
Entity(
  id,
  canonical_name TEXT,
  type TEXT,                 -- e.g. 'Model', 'Dataset', 'Concept'
  normalized_name TEXT,      -- lowercase, stripped version for merging
  extra_metadata JSONB       -- optional (aliases, etc.)
)

EntityMention(
  id,
  entity_id REFERENCES Entity,
  sentence_id REFERENCES Sentence,
  char_start INT,
  char_end INT,
  surface_text TEXT
)
```

### Claims & relations

```sql
ClaimSentence(
  id,
  sentence_id REFERENCES Sentence,
  is_claim BOOLEAN,
  score REAL                -- optional, how "claim-like" it is
)

Relation(
  id,
  head_entity_id REFERENCES Entity,
  tail_entity_id REFERENCES Entity,
  relation_type TEXT,       -- 'improves', 'evaluated_on', 'is_a', 'related_to', ...
  confidence REAL,
  created_at TIMESTAMPTZ
)

RelationEvidence(
  id,
  relation_id REFERENCES Relation,
  sentence_id REFERENCES Sentence,
  explanation TEXT,         -- optional snippet or pattern name
  created_at TIMESTAMPTZ
)
```

You might also want a `RelationType` table if you want stricter schema:

```sql
RelationType(
  name TEXT PRIMARY KEY,      -- 'improves', 'evaluated_on', ...
  allowed_head_types TEXT[],  -- ARRAY of entity types
  allowed_tail_types TEXT[]   -- ARRAY of entity types
)
```

This lets you enforce type constraints in SQL and in code.

---

## 4. How this hits your original three goals

### 1) Research tool

* Entities → browsable index of concepts.
* Claims → show “what is asserted about X”, backed by evidence sentences.
* Relations → see how models/datasets/etc. connect.
* Provenance → every edge is clickable back to the original text.

### 2) Reasoning substrate for agents

* Agents can query: “graph around X” and get a **typed subgraph** + evidence to ground their reasoning.
* You avoid hallucinated connections because every edge is evidence-linked.
* You can give agents explicit structure:

  * `improves(ModelA, MetricB) with confidence=0.9, evidence=...`

### 3) Open-source alternative to opaque vector stores

* You still support standard vector search (chunks embeddings).
* BUT you also expose:

  * Entities and relations as first-class, explainable objects.
  * Schema and storage that are transparent (Postgres + pgvector).
* This is something LangChain-like vector stores **don’t** give you:
  they store dense blobs, not interpretable graph structure.

---

## 5. What to implement *next*, concretely (MVP path)

So this doesn’t become another multi-week detour, I’d do it in this order:

### Milestone 1 – Sentences + entities

1. Add `Sentence`, `Entity`, `EntityMention` tables.
2. For each existing `Chunk`:

   * Split into `Sentence`s using spaCy.
   * Run NER to fill `Entity` + `EntityMention`.
3. Build a tiny API:

   * `GET /entities?q=...` (search by name, return linked sentences & docs).

You now have typed entities and evidence, even before relations.

---

### Milestone 2 – Claims + basic relations

1. Add `ClaimSentence`, `Relation`, `RelationEvidence`.
2. Implement maybe **3 relation types** to start:

   * `is_a`
   * `evaluated_on`
   * `improves`
3. Implement dependency-based patterns just for these.
4. Apply type constraints via a small in-code matrix (or `RelationType` table).

Now you have a small, **high-precision** relation graph you can inspect.

---

### Milestone 3 – Productize for UI / agents

1. Update your API to support:

   * `GET /entity/{id}/neighbors`
   * `GET /relation/{id}` (with evidence)
2. Wire your UI:

   * Entity page: show type, description, claims, relations.
   * Graph view: only show high-confidence typed edges by default (toggle `related_to` separately).
3. Agent integration:

   * Add a `/graph_context` endpoint that returns: entities, relations, evidence sentences.

---

### Bottom line

Your original ambition is still viable. The bottleneck was the extraction strategy, not the goal.

This pipeline:

* Uses **off-the-shelf ML only** (NER + parser).
* Keeps your code **structured and composable** (clear stages, clear tables).
* Gives you a **clean graph** with evidence and type constraints.
* Is absolutely doable by a single person over a few focused iterations.

If you want, next step I can do is:
take one of your existing “garbage” relation CSV rows and walk through exactly how it would be produced (or *rejected*) by this pipeline, so you can see the difference in practice.

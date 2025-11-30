# Pipeline stages (revised, realistic version)

### Stage 0 – Ingestion & chunking (keep what you have)

You likely already do something like:

* `Document` row per file/url/etc.
* Split into `Chunk`s (e.g., ~500–1000 tokens) with overlap.
* Embed chunks → pgvector column `embedding`.

Keep this. Maybe just tighten sentence splitting later (but that’s not your main bottleneck).

---

### Stage 1 – Sentence splitting per chunk

Goal: each **claim** or **relation** should hang off a reasonably clean sentence.

* For each `Chunk`:

  * Run a robust sentence splitter (e.g., spaCy `doc.sents` instead of regex).
  * Store sentences in a `Sentence` table.

**Key point:**
Stop trying to get claims directly from raw chunk text. Claims live at the sentence level.

---

### Stage 2 – NER + entity typing (typed entities)

For each sentence:

1. Run NER and parsing (spaCy model or similar, all local, no training).
2. Extract **entity mentions** (span + label).
3. Map raw NER labels → your ontology:

Example mapping:

* `ORG` → `Organization`
* `PERSON` → `Person`
* `GPE`, `LOC` → `Location`
* `PRODUCT`, `WORK_OF_ART` → maybe `Model` or `Method` depending on context; if unsure → `Concept`
* Everything else → `Concept` (safe default)

4. Normalize entities:

   * Lowercase, strip punctuation.
   * Simple heuristics or fuzzy match to merge repeating entities across docs later (keep it simple at first).

This produces:

* `Entity` (canonical node: “GPT-4”, “ImageNet”, “Google”)
* `EntityMention` (this occurrence of that entity in this sentence, with char offsets, etc.)

**Important:** Entity typing is not another “filter”; it’s the schema that will constrain relations.

---

### Stage 3 – Claim sentence detection (optional but useful)

You can either:

* Start simple: “all sentences with ≥ 2 entities and a verb are claim candidates”.
* Or add a tiny classifier later (LLM or local model) to flag “is this a claim-like sentence? (yes/no)”.

For now, given constraints, do **simple deterministic rules**:

Mark a sentence as a `ClaimSentence` if:

* It contains at least 2 entity mentions, **and**
* Contains a verb that’s in one of your “relation verbs” lists (like `is, improves, increases, depends on, uses, trained on, evaluated on, causes` etc.).

This is coarse, but focusing only on “claim-like verbs + ≥2 entities” already cuts noise.

**Implementation notes:**

* Implemented in `okc_pipeline.stage_03_claims.claim_detector`.
* Persists to the `claim_sentence` table with a coarse `score` and metadata about the verbs/hedges hit.
* Enabled by adding a `claims` stage entry in `pipeline.yaml`.

---

### Stage 4 – Relation candidate extraction (dependency-driven, not regex-driven)

For each `ClaimSentence`:

1. Use the **dependency parse** to find basic structures:

   * Subject (`nsubj`), object (`dobj`, `attr`, `pobj`), main verb (`ROOT` or key verbs).
2. Implement a small set of **relation templates**, something like:

**Examples:**

1. Definition / IS_A

   * Pattern: `X is (a|an|the) Y`
   * Relation type: `is_a(X:Concept, Y:Concept)`

2. Improves / Degrades

   * Verbs: improves, increases, boosts, reduces error, outperforms
   * Relation type: `improves(X:Model/Method, Y:Metric)`

3. Evaluated On

   * Patterns: `X on Y`, `X evaluated on Y`, `X tested on Y`
   * Relation type: `evaluated_on(X:Model/Method, Y:Dataset)`

4. Depends On / Requires

   * Pattern: `X depends on Y`, `X requires Y`
   * Relation type: `depends_on(X, Y)`

5. Part-of / Subcomponent

   * Pattern: `X is part of Y`, `X component of Y`
   * Relation type: `part_of(X:Concept, Y:Concept)`

A **relation candidate** is `(entity1, relation_type, entity2, sentence)` that matches:

* A dependency pattern, **and**
* Type constraints (see next stage).

Everything else (co-occurring entities in the same sentence that don’t match a pattern) can optionally become a **weak `related_to` edge**.

---

### Stage 5 – Type constraints & scoring (avoid junk)

Now you use typed entities to **filter** relations:

For each candidate:

* Check `(type(entity1), type(entity2), relation_type)` against a small allowed matrix.

Example:

* `improves`: allowed pairs =

  * `(Model, Metric)`, `(Method, Metric)`
* `evaluated_on`: `(Model|Method, Dataset)`
* `is_a`: `(Concept, Concept)`
* `depends_on`: any types allowed (broad)

If a candidate fails its type constraint, either:

* Drop it (for strict types like `improves`, `evaluated_on`), or
* Downgrade it to `related_to` with lower confidence.

Assign a **confidence score**:

* +1 if type pair is allowed
* +1 if the verb is exact match in your relation verb list
* +1 if dependency pattern is clear (subject/verb/object clean)
* -1 if sentence contains hedging (`might`, `could`, `suggests that`)

This doesn’t need to be fancy, just enough to separate “likely correct” from “iffy”.

Store `confidence` as a float/int in the table.

---

### Stage 6 – Persist to graph tables (claims, relations, evidence)

At this point, you’ve got:

* sentences
* entities & mentions
* relation candidates with types & confidence

Persist as:

* `ClaimSentence` (points to `Sentence`, has a boolean / score “is_claim”)
* `Relation` (structured edge between canonical `Entity` nodes)
* `RelationEvidence` (many-to-one: multiple pieces of evidence per relation, each pointing to a `Sentence`)

This gives you:

* Machine-usable graph (`Entity`, `Relation`)
* Human-auditable evidence (`Sentence`, `RelationEvidence`)

---

### Stage 7 – Query & API surface

Now expose this in your API (FastAPI):

1. **Entity search**

   * Input: text; use vector search to find relevant chunks, then surface entities in those chunks.
   * Or text → embed → find best entities by like “name similarity + co-mentions”.

2. **Explain a relation**

   * Given `entity1`, `entity2`, show:

     * relation types between them
     * confidence
     * list of evidence sentences (with links back to document)

3. **Graph neighbors**

   * Given an entity, show:

     * typed edges (tabs by relation_type, like “improves”, “evaluated_on”, “is_a”)
     * generic `related_to` edges separately so they don’t pollute core structure.

4. **Research queries**

   * “What models improve accuracy on dataset X?”
   * Implementation:

     * Find entity for dataset X.
     * Graph query: relations where `relation_type='evaluated_on' AND entity2 = X`.
     * Join to `improves` relations going from those models to metrics.

5. **Agent grounding**

   * Agent gets:

     * A subgraph (entities + relations + evidence sentences) returned as JSON.
     * That becomes its “ground truth context” instead of raw chunks only.

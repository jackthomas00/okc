## 1. Overall architecture: what OKC *is*

Think of OKC as **two tightly-linked layers** over the same corpus:

1. **Vector layer (for recall)**

   * Chunks with embeddings (what you already have).
   * Used to “find relevant text” fast.

2. **Graph layer (for structure + reasoning)**

   * Entities (typed)
   * Entity mentions (where they appear)
   * Relations (typed or generic)
   * Claims / evidence (sentences tied to relations)

All grounded in **provenance**: every structured thing points back to a specific chunk/sentence in a specific document.

You already have layer 1. The redesign is basically about making layer 2 sane and maintainable.

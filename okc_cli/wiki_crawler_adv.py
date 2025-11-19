#!/usr/bin/env python3
import os, json, time, random, hashlib, sys, signal
from collections import deque
from typing import List, Tuple, Optional

import requests
import numpy as np

# Import shared filter logic
from pipeline.ingestion.doc_filters import keep_doc  # noqa: E402

# ======= CONFIG =======
OKC_API = "http://localhost:8000/ingest_bulk"   # your OKC /ingest_bulk
USER_AGENT = "OpenKnowledgeCompiler/0.3 (https://github.com/jackthomas00/okc; contact: jack@jack-thomas.com)"
SLEEP_BASE = 0.30           # seconds base between requests (jitter added)
BATCH_SIZE = 50             # /ingest_bulk payload size
MAX_PAGES = 8000            # hard cap for accepted docs
MAX_HOPS = 2                # BFS depth (0 = seeds only)
MAX_FANOUT = 400            # max links per page to enqueue
SEED_TITLES = [
    "Transformer (machine learning)", "Self-attention", "BERT (language model)",
    "Retrieval-augmented generation", "Word embedding", "Vector search",
    "Graph database", "Knowledge graph", "BM25", "Natural language processing",
    "Neural network", "Reinforcement learning", "Diffusion model"
]
SEED_CATEGORIES = [  # optional category seeds; use plain names without "Category:" prefix
    "Natural language processing",
    "Machine learning",
    "Information retrieval"
]

# Approximate dedupe
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
COSINE_DUP_THRESHOLD = 0.95                           # consider duplicate if >= this
EMB_BATCH = 32                                        # batch size for embedding
SAVE_EVERY = 300                                      # save checkpoint every N accepted docs
CHECKPOINT_PATH = "okc_wiki_state.json"
EMBEDS_PATH = "okc_wiki_embeds.npy"                   # persisted embedding matrix
META_PATH = "okc_wiki_meta.json"                      # maps row -> {title, url, hash}

# Timeouts/retries
REQ_TIMEOUT = 20
RETRY_AFTER_FALLBACK = 5

# ======================

# Endpoints
REST_PLAIN = "https://en.wikipedia.org/api/rest_v1/page/plain/{}"
ACTION_API = "https://en.wikipedia.org/w/api.php"

# Session with polite headers
S = requests.Session()
S.headers.update({"User-Agent": USER_AGENT})

def sleep_politely():
    time.sleep(SLEEP_BASE + random.random() * 0.35)

def backoff(resp):
    wait = int(resp.headers.get("Retry-After", RETRY_AFTER_FALLBACK))
    time.sleep(wait + random.random() * 1.5)

def normalize_title(t: str) -> str:
    return t.strip().replace(" ", "_")

def pretty_title(t: str) -> str:
    return t.replace("_", " ")

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def fetch_plain(title: str) -> Optional[str]:
    url = REST_PLAIN.format(requests.utils.quote(title))
    while True:
        try:
            r = S.get(url, timeout=REQ_TIMEOUT)
            if r.status_code == 200:
                return r.text.strip()
            if r.status_code in (429, 503):
                backoff(r); continue
            return None
        except requests.RequestException:
            time.sleep(1.0); continue

def get_links(title: str) -> List[str]:
    """Return article titles linked from this page (namespace 0 only)."""
    links = []
    params = {
        "action": "query", "prop": "links", "format": "json",
        "titles": title, "plnamespace": 0, "pllimit": "max"
    }
    while True:
        try:
            r = S.get(ACTION_API, params=params, timeout=REQ_TIMEOUT)
            if r.status_code in (429, 503):
                backoff(r); continue
            r.raise_for_status()
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for _, page in pages.items():
                for l in page.get("links", []):
                    link_title = l["title"]
                    if ":" in link_title:
                        continue
                    links.append(normalize_title(link_title))
            if "continue" in data:
                params.update(data["continue"])
                sleep_politely()
                continue
            return links
        except requests.RequestException:
            time.sleep(1.0); continue

def get_category_members(category: str, limit_per_category: int = 2000) -> List[str]:
    """Return article titles from Category:category (namespace 0)."""
    titles = []
    params = {
        "action": "query", "list": "categorymembers", "format": "json",
        "cmtitle": f"Category:{category}", "cmnamespace": 0, "cmlimit": "max"
    }
    while True:
        try:
            r = S.get(ACTION_API, params=params, timeout=REQ_TIMEOUT)
            if r.status_code in (429, 503):
                backoff(r); continue
            r.raise_for_status()
            data = r.json()
            for it in data.get("query", {}).get("categorymembers", []):
                t = it.get("title")
                if not t: continue
                if ":" in t: continue
                titles.append(normalize_title(t))
                if len(titles) >= limit_per_category:
                    return titles
            if "continue" in data:
                params.update(data["continue"])
                sleep_politely()
                continue
            return titles
        except requests.RequestException:
            time.sleep(1.0); continue

def clean_text(text: str) -> str:
    lowers = text.lower()
    cutpoints = ["\nreferences\n", "\nexternal links\n", "\nsee also\n", "\nfurther reading\n"]
    idxs = [lowers.find(k) for k in cutpoints if lowers.find(k) != -1]
    if idxs:
        text = text[:min(idxs)]
    return " ".join(text.split())

# keep_doc is imported from backend.pipeline.ingestion.doc_filters

# ----- Embeddings / Dedupe -----
try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

_model = None
EMB_DIM = 384  # all-MiniLM-L6-v2
def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMB_MODEL)
    return _model

def embed_texts(txts: List[str]) -> np.ndarray:
    model = get_model()
    vecs = model.encode(txts, normalize_embeddings=True, convert_to_numpy=True)
    if vecs.ndim == 1: vecs = vecs.reshape(1, -1)
    return vecs.astype(np.float32)

class ANNIndex:
    """Approximate (FAISS) or exact (numpy) cosine similarity index."""
    def __init__(self, dim: int, use_faiss: bool):
        self.dim = dim
        self.use_faiss = use_faiss and HAVE_FAISS
        self.count = 0
        self._index = None
        self._vecs = None  # numpy fallback
        if self.use_faiss:
            # FAISS Index for inner product (cosine with normalized vectors)
            self._index = faiss.IndexFlatIP(dim)

    def add(self, vecs: np.ndarray):
        if self.use_faiss:
            self._index.add(vecs)
        else:
            self._vecs = vecs if self._vecs is None else np.vstack([self._vecs, vecs])
        self.count += vecs.shape[0]

    def search(self, vecs: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            # no data yet -> return -inf score
            return np.full((vecs.shape[0], k), -1, dtype=np.int64), np.full((vecs.shape[0], k), -1.0, dtype=np.float32)
        if self.use_faiss:
            scores, idxs = self._index.search(vecs, k)
            # faiss returns scores first for IP; align to (idxs, scores)
            return idxs, scores
        else:
            # brute-force cosine (vecs are normalized)
            sims = vecs @ self._vecs.T  # (q, n)
            idxs = np.argmax(sims, axis=1).reshape(-1, 1)
            scores = np.take_along_axis(sims, idxs, axis=1)
            return idxs.astype(np.int64), scores.astype(np.float32)

    def save_numpy(self, path: str):
        if self.use_faiss:
            # dump faiss index to bytes via write_index
            faiss.write_index(self._index, path + ".faiss")
        else:
            if self._vecs is None:
                np.save(path, np.empty((0, self.dim), dtype=np.float32))
            else:
                np.save(path, self._vecs)

    def load_numpy(self, path: str):
        if self.use_faiss:
            p = path + ".faiss"
            if os.path.exists(p):
                self._index = faiss.read_index(p)
                self.count = self._index.ntotal
        else:
            if os.path.exists(path):
                self._vecs = np.load(path)
                self.count = self._vecs.shape[0]

# ----- Checkpointing -----
def save_state(state: dict, ann: ANNIndex, meta: dict):
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, CHECKPOINT_PATH)

    # embeds + meta
    ann.save_numpy(EMBEDS_PATH)
    with open(META_PATH + ".tmp", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    os.replace(META_PATH + ".tmp", META_PATH)

def load_state() -> Tuple[dict, ANNIndex, dict]:
    state = {
        "queue": [],              # list of [title, depth]
        "visited_titles": [],     # normalized titles
        "seen_hashes": [],        # sha1 of normalized text
        "ingested": 0
    }
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            pass
    ann = ANNIndex(EMB_DIM, use_faiss=True)
    ann.load_numpy(EMBEDS_PATH)

    meta = {}
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    return state, ann, meta

# ----- Posting -----
def post_batch(batch: List[dict]) -> List[dict]:
    while True:
        try:
            resp = S.post(OKC_API, json={"items": batch}, timeout=180)
            if resp.status_code in (429, 503):
                backoff(resp); continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            time.sleep(2.0); continue

# ----- Main crawl -----
def crawl():
    # Load or init state
    state, ann, meta = load_state()
    visited_titles = set(state.get("visited_titles", []))
    seen_hashes = set(state.get("seen_hashes", []))
    ingested = int(state.get("ingested", 0))

    q = deque(state.get("queue", []))
    if not q:
        # seed queue from titles + categories
        for t in SEED_TITLES:
            q.append([normalize_title(t), 0])
        for cat in SEED_CATEGORIES:
            for t in get_category_members(cat, limit_per_category=2000):
                q.append([t, 0])

    batch: List[dict] = []
    approx_emb_buf: List[np.ndarray] = []
    approx_meta_buf: List[dict] = []  # keeps (title,url,hash) for new accepted docs in this session

    def snapshot():
        # serialize queue
        state["queue"] = list(q)
        state["visited_titles"] = list(visited_titles)
        state["seen_hashes"] = list(seen_hashes)
        state["ingested"] = ingested
        save_state(state, ann, meta)

    # graceful stop
    stop_flag = {"stop": False}
    def handle_sigint(sig, frame):
        stop_flag["stop"] = True
        print("\n[!] Caught interrupt; saving checkpoint...")
        snapshot()
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    last_save_ingested = ingested

    while q and ingested < MAX_PAGES:
        title, depth = q.popleft()
        if title in visited_titles:
            continue
        visited_titles.add(title)

        txt = fetch_plain(title)
        sleep_politely()
        if not txt:
            continue
        txt = clean_text(txt)
        if not keep_doc(txt, title=title):
            continue
        h = sha1(txt)
        if h in seen_hashes:
            continue

        # Approximate dedupe via embedding
        # Build a short doc-level representation: take the first ~1200 chars (often enough)
        head = txt[:1200]
        vec = embed_texts([head])[0].reshape(1, -1)
        if ann.count > 0:
            idxs, scores = ann.search(vec, k=1)
            if scores[0, 0] >= COSINE_DUP_THRESHOLD:
                # likely near-duplicate; skip
                continue

        # Accept locally
        seen_hashes.add(h)
        url = f"https://en.wikipedia.org/wiki/{title}"
        batch.append({
            "title": pretty_title(title),
            "url": url,
            "text": txt
        })
        approx_emb_buf.append(vec)
        approx_meta_buf.append({"title": pretty_title(title), "url": url, "hash": h})

        # Expand links
        if depth < MAX_HOPS:
            try:
                links = get_links(title)
                sleep_politely()
                random.shuffle(links)
                for lt in links[:MAX_FANOUT]:
                    if lt not in visited_titles:
                        q.append([lt, depth + 1])
            except Exception:
                pass

        # Flush if needed
        if len(batch) >= BATCH_SIZE or (not q and batch):
            results = post_batch(batch)
            # Update counters + ANN only for accepted docs (not deduped by server)
            accepted = 0
            new_vecs = []
            for it, res, emb_meta, v in zip(batch, results, approx_meta_buf, approx_emb_buf):
                if not res.get("deduped"):
                    accepted += 1
                    new_vecs.append(v)
                    # record in meta with sequential index
                    meta[str(ann.count + len(new_vecs) - 1)] = emb_meta
            if new_vecs:
                ann.add(np.vstack(new_vecs))
            ingested += accepted
            print(f"[OKC] pushed {len(batch)}, accepted {accepted}, total={ingested}, ann_size={ann.count}")

            # clear buffers
            batch.clear()
            approx_emb_buf.clear()
            approx_meta_buf.clear()

            # periodic checkpoint
            if ingested - last_save_ingested >= SAVE_EVERY:
                snapshot()
                last_save_ingested = ingested

        if stop_flag["stop"] or ingested >= MAX_PAGES:
            break

    # final snapshot
    snapshot()
    print(f"Done. Total accepted: {ingested}, visited: {len(visited_titles)}, ann_size={ann.count}")

if __name__ == "__main__":
    crawl()

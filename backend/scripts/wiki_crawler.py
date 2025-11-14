#!/usr/bin/env python3
import requests, time, random, hashlib, json, sys
from collections import deque
from pathlib import Path

# Import shared filter logic
from pipeline.ingestion.doc_filters import keep_doc  # noqa: E402

# ======= CONFIG =======
OKC_API = "http://localhost:8000/ingest_bulk"     # your OKC API
USER_AGENT = "OpenKnowledgeCompiler/0.1 (https://github.com/jackthomas00/okc; contact: jack@jack-thomas.com)"
SLEEP_BASE = 0.30          # seconds between requests (will add jitter)
BATCH_SIZE = 50            # /ingest_bulk payload size
MAX_PAGES = 8000           # hard cap
MAX_HOPS = 2               # BFS depth from seeds (0 = seeds only)
SEED_TITLES = [
    "Transformer (machine learning)", "Self-attention", "BERT (language model)",
    "Retrieval-augmented generation", "Word embedding", "Vector search",
    "Graph database", "Knowledge graph", "BM25", "Natural language processing",
    "Neural network", "Reinforcement learning", "Diffusion model"
]
# ======================

ACTION_API  = "https://en.wikipedia.org/w/api.php"

S = requests.Session()
S.headers.update({"User-Agent": USER_AGENT})

def sleep_politely():
    time.sleep(SLEEP_BASE + random.random() * 0.35)

def backoff(resp):
    if resp.status_code in (429, 503):
        wait = int(resp.headers.get("Retry-After", 5))
        time.sleep(wait + random.random() * 2)

def normalize_title(t: str) -> str:
    return t.strip().replace(" ", "_")

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def fetch_plain(title: str) -> str | None:
    """Fetch plain text content of a Wikipedia article using Action API."""
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": False,  # get full article, not just intro
        "explaintext": True,  # plain text, no HTML
        "format": "json",
        "titles": title,
        "redirects": 1  # follow redirects
    }
    while True:
        try:
            r = S.get(ACTION_API, params=params, timeout=20)
            if r.status_code in (429, 503):
                backoff(r); continue
            r.raise_for_status()
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                # skip missing pages
                if page.get("missing"):
                    return None
                extract = page.get("extract", "").strip()
                if extract:
                    return extract
            return None
        except requests.RequestException:
            time.sleep(1.5)
            continue

def get_links(title: str) -> list[str]:
    """Return article titles linked from this page (namespace 0 only)."""
    links = []
    params = {
        "action": "query", "prop": "links", "format": "json",
        "titles": title, "plnamespace": 0, "pllimit": "max"
    }
    while True:
        try:
            r = S.get(ACTION_API, params=params, timeout=20)
            if r.status_code in (429, 503):
                backoff(r); continue
            r.raise_for_status()
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for _, page in pages.items():
                for l in page.get("links", []):
                    link_title = l["title"]
                    if ":" in link_title:  # filter non-article namespaces defensively
                        continue
                    links.append(normalize_title(link_title))
            if "continue" in data:
                params.update(data["continue"])
                sleep_politely()
                continue
            return links
        except requests.RequestException:
            time.sleep(1.5)
            continue

def clean_text(text: str) -> str:
    # Stop at common end sections; keep headings/paras before that.
    lowers = text.lower()
    cutpoints = ["\nreferences\n", "\nexternal links\n", "\nsee also\n", "\nfurther reading\n"]
    idxs = [lowers.find(k) for k in cutpoints if lowers.find(k) != -1]
    if idxs:
        text = text[:min(idxs)]
    # collapse weird whitespace
    return " ".join(text.split())

# keep_doc is imported from backend.pipeline.ingestion.doc_filters

def post_batch(batch: list[dict]) -> list[dict]:
    while True:
        try:
            resp = S.post(OKC_API, json={"items": batch}, timeout=180)
            if resp.status_code in (429, 503):
                backoff(resp); continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            time.sleep(2.0)
            continue

def crawl():
    visited_titles: set[str] = set()
    seen_hashes: set[str] = set()
    q = deque((normalize_title(t), 0) for t in SEED_TITLES)

    batch: list[dict] = []
    ingested = 0

    while q and ingested < MAX_PAGES:
        title, depth = q.popleft()
        if title in visited_titles:
            continue
        visited_titles.add(title)

        # fetch text
        txt = fetch_plain(title)
        sleep_politely()
        if not txt:
            print(f"[SKIP] {title}: empty text")
            continue
        txt = clean_text(txt)
        if not keep_doc(txt, title=title):
            print(f"[SKIP] {title}: doesn't match filter criteria")
            continue
        h = sha1(txt)
        if h in seen_hashes:
            print(f"[SKIP] {title}: duplicate hash (client-side)")
            continue
        seen_hashes.add(h)

        # stage for OKC
        batch.append({
            "title": title.replace("_", " "),
            "url": f"https://en.wikipedia.org/wiki/{title}",
            "text": txt
        })
        print(f"[ADD] {title} to batch (batch size: {len(batch)})")

        # expand links (BFS) if we have depth budget
        if depth < MAX_HOPS:
            try:
                links = get_links(title)
                sleep_politely()
                random.shuffle(links)               # mix topics
                for lt in links[:400]:              # cap fan-out per node
                    if lt not in visited_titles:
                        q.append((lt, depth + 1))
            except Exception:
                pass

        # flush batches
        if len(batch) >= BATCH_SIZE or (not q and batch):
            print(f"[POST] Sending batch of {len(batch)} items to API...")
            results = post_batch(batch)
            print(f"[DEBUG] API returned {len(results)} results")
            for i, r in enumerate(results[:3]):  # show first 3 results
                print(f"  Result {i}: deduped={r.get('deduped')}, title={r.get('title')}")
            ok = sum(1 for r in results if not r.get("deduped"))
            ingested += ok
            print(f"[OKC] pushed {len(batch)}, accepted {ok}, total={ingested}")
            batch.clear()

        # hard stop
        if ingested >= MAX_PAGES:
            break

    print(f"Done. Total accepted: {ingested}, visited: {len(visited_titles)}")

if __name__ == "__main__":
    try:
        crawl()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)

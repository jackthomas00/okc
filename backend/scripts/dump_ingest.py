#!/usr/bin/env python3
import os, json, hashlib, glob, time, sys
import requests
from typing import Iterable

# Import shared filter logic
from pipeline.ingestion.doc_filters import keep_doc  # noqa: E402

# ========= CONFIG =========
OKC_API = "http://localhost:8000/ingest_bulk"   # your OKC endpoint
EXTRACT_DIR = "extracted"                       # where WikiExtractor wrote JSON files
CHECKPOINT = "dump_ingest_state.json"           # resume progress here
BATCH_SIZE = 50                                 # /ingest_bulk payload size
# ==========================

S = requests.Session()
S.headers.update({
    "User-Agent": "OpenKnowledgeCompiler/0.3 (https://github.com/jackthomas00/okc; contact: jack@jack-thomas.com)"
})

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def iter_json_files(root: str) -> Iterable[str]:
    # WikiExtractor creates nested dirs; stream deterministically
    # WikiExtractor outputs files named wiki_XX (no extension) or *.json
    json_patterns = [
        os.path.join(root, "**", "*.json"),
        os.path.join(root, "**", "wiki_*")
    ]
    seen = set()
    for pattern in json_patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            if path not in seen and os.path.isfile(path):
                seen.add(path)
                yield path

def iter_json_lines(path: str, start_offset: int = 0) -> Iterable[tuple[int, dict]]:
    with open(path, "r", encoding="utf-8") as f:
        if start_offset:
            f.seek(start_offset)
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                yield offset, json.loads(line)
            except json.JSONDecodeError:
                continue

# keep_doc is imported from backend.pipeline.ingestion.doc_filters

def load_state():
    if not os.path.exists(CHECKPOINT):
        return {
            "files": [],         # list of processed files (complete)
            "current_file": "",  # path currently processing
            "offset": 0,         # byte offset within current file
            "seen_hashes": [],   # dedupe within this run
            "accepted": 0,       # count of accepted (server non-deduped) docs
            "posted": 0          # total posted (including dedup hits)
        }
    with open(CHECKPOINT, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state):
    tmp = CHECKPOINT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, CHECKPOINT)

def post_batch(batch):
    # retries with small backoff
    backoff = 2
    while True:
        try:
            r = S.post(OKC_API, json={"items": batch}, timeout=300)
            if r.status_code in (429, 503):
                wait = int(r.headers.get("Retry-After", 5))
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

def run():
    state = load_state()
    seen_hashes = set(state.get("seen_hashes", []))
    batch = []

    files = list(iter_json_files(EXTRACT_DIR))
    # resume logic
    file_iter = files
    if state["current_file"] and state["current_file"] in files:
        # start from current_file then continue with remaining
        idx = files.index(state["current_file"])
        file_iter = files[idx:]
    elif state["files"]:
        # start after last fully processed file
        done_set = set(state["files"])
        file_iter = [p for p in files if p not in done_set]

    for path in file_iter:
        start_offset = state["offset"] if path == state.get("current_file") else 0
        state["current_file"] = path
        state["offset"] = 0
        state["seen_hashes"] = list(seen_hashes)
        save_state(state)

        processed_in_file = 0
        for offset, doc in iter_json_lines(path, start_offset=start_offset):
            state["offset"] = offset
            # minimal schema from WikiExtractor
            title = doc.get("title", "").strip()
            url = doc.get("url") or (f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" if title else None)
            text = doc.get("text", "").strip()
            if not title or not text:
                continue

            if not keep_doc(text, title=title):
                continue

            h = sha1(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            batch.append({"title": title, "url": url, "text": text})
            processed_in_file += 1

            # flush
            if len(batch) >= BATCH_SIZE:
                results = post_batch(batch)
                accepted = sum(1 for r in results if not r.get("deduped"))
                state["accepted"] += accepted
                state["posted"] += len(batch)
                print(f"[OKC] file={os.path.basename(path)} posted={len(batch)} accepted={accepted} total_accepted={state['accepted']}")
                batch.clear()
                state["seen_hashes"] = list(seen_hashes)
                save_state(state)

        # end-of-file flush
        if batch:
            results = post_batch(batch)
            accepted = sum(1 for r in results if not r.get("deduped"))
            state["accepted"] += accepted
            state["posted"] += len(batch)
            print(f"[OKC] file={os.path.basename(path)} posted={len(batch)} accepted={accepted} total_accepted={state['accepted']}")
            batch.clear()
            state["seen_hashes"] = list(seen_hashes)
            save_state(state)

        # mark file complete
        state["files"].append(path)
        state["current_file"] = ""
        state["offset"] = 0
        state["seen_hashes"] = list(seen_hashes)
        save_state(state)

    print(f"Done. Accepted={state['accepted']} Posted={state['posted']} FilesDone={len(state['files'])}")

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterrupted; state saved.")
        sys.exit(0)

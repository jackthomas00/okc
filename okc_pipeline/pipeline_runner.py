#!/usr/bin/env python3
"""
Pipeline runner that processes YAML configuration files.

Supports multiple source types:
- wikipedia_api: Uses Wikipedia API crawler
- extracted_folder: Uses extracted folder (dump ingest)

Processes stages in sequence: chunk, embed, entities, topics, claims, relations
"""

import os
import sys
import json
import yaml
import hashlib
import time
import random
import glob
import argparse
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List
from collections import deque

import requests
import numpy as np

# Note: We'll import crud and embedder modules after setting environment variables
# in the PipelineRunner.__init__ method to ensure the correct model is used


# ========== Wikipedia API Source ==========

ACTION_API = "https://en.wikipedia.org/w/api.php"
REST_PLAIN = "https://en.wikipedia.org/api/rest_v1/page/plain/{}"
USER_AGENT = "OpenKnowledgeCompiler/0.3 (https://github.com/jackthomas00/okc; contact: jack@jack-thomas.com)"

S = requests.Session()
S.headers.update({"User-Agent": USER_AGENT})


def sleep_politely(base: float = 0.30):
    time.sleep(base + random.random() * 0.35)


def backoff(resp: requests.Response):
    if resp.status_code in (429, 503):
        wait = int(resp.headers.get("Retry-After", 5))
        time.sleep(wait + random.random() * 1.5)


def normalize_title(t: str) -> str:
    return t.strip().replace(" ", "_")


def fetch_wikipedia_article(title: str, domain: str = "en") -> Optional[str]:
    """Fetch plain text content of a Wikipedia article."""
    if domain != "en":
        # For non-English, use Action API
        url = f"https://{domain}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": False,
            "explaintext": True,
            "format": "json",
            "titles": title,
            "redirects": 1
        }
        while True:
            try:
                r = S.get(url, params=params, timeout=20)
                if r.status_code in (429, 503):
                    backoff(r)
                    continue
                r.raise_for_status()
                data = r.json()
                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    if page.get("missing"):
                        return None
                    extract = page.get("extract", "").strip()
                    if extract:
                        return extract
                return None
            except requests.RequestException:
                time.sleep(1.5)
                continue
    else:
        # Use REST API for English (faster)
        url = REST_PLAIN.format(requests.utils.quote(title))
        while True:
            try:
                r = S.get(url, timeout=20)
                if r.status_code == 200:
                    return r.text.strip()
                if r.status_code in (429, 503):
                    backoff(r)
                    continue
                return None
            except requests.RequestException:
                time.sleep(1.0)
                continue


def get_wikipedia_links(title: str, domain: str = "en") -> List[str]:
    """Return article titles linked from this page."""
    url = f"https://{domain}.wikipedia.org/w/api.php"
    links = []
    params = {
        "action": "query",
        "prop": "links",
        "format": "json",
        "titles": title,
        "plnamespace": 0,
        "pllimit": "max"
    }
    while True:
        try:
            r = S.get(url, params=params, timeout=20)
            if r.status_code in (429, 503):
                backoff(r)
                continue
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
            time.sleep(1.0)
            continue


def clean_text(text: str) -> str:
    """Clean Wikipedia text."""
    lowers = text.lower()
    cutpoints = ["\nreferences\n", "\nexternal links\n", "\nsee also\n", "\nfurther reading\n"]
    idxs = [lowers.find(k) for k in cutpoints if lowers.find(k) != -1]
    if idxs:
        text = text[:min(idxs)]
    return " ".join(text.split())


# ========== Extracted Folder Source ==========

def iter_json_files(root: str) -> Iterable[str]:
    """Iterate over JSON files in extracted folder."""
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
    """Iterate over JSON lines in a file."""
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


# ========== Pipeline Runner ==========

class PipelineRunner:
    """Runs the pipeline based on YAML configuration."""
    
    def __init__(self, config_path: str, skip_stages: Optional[List[str]] = None, skip_sourcing: bool = False, clear_db: bool = False):
        """
        Initialize pipeline runner with YAML config.
        
        Args:
            config_path: Path to YAML configuration file
            skip_stages: Optional list of stage names to skip (e.g., ["entities", "topics"])
            skip_sourcing: If True, skip document sourcing and process existing data from database
            clear_db: If True, clear all documents from database before running (cascades to all downstream data)
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.sources = self.config.get("sources", [])
        all_stages = self.config.get("stages", [])
        self.skip_sourcing = skip_sourcing
        self.clear_db = clear_db
        
        # Filter out skipped stages
        if skip_stages:
            skip_set = set(skip_stages)
            self.stages = [s for s in all_stages if s.get("name") not in skip_set]
            skipped = [s.get("name") for s in all_stages if s.get("name") in skip_set]
            if skipped:
                print(f"[Pipeline] Skipping stages: {', '.join(skipped)}")
        else:
            self.stages = all_stages
        
        # Extract global config
        global_config = self.config.get("config", {})
        self.spacy_model = global_config.get("spacy_model")
        
        # Extract stage configurations
        self.chunk_params = {}
        self.embed_model = None
        self.enable_sentences = False
        self.enable_entities = False
        self.enable_claims = False
        self.enable_relations = False
        
        for stage in self.stages:
            if stage.get("name") == "chunk":
                self.chunk_params = stage.get("params", {})
            elif stage.get("name") == "embed":
                self.embed_model = stage.get("model", "sentence-transformers/all-MiniLM-L6-v2")
            elif stage.get("name") == "sentences":
                self.enable_sentences = True
            elif stage.get("name") == "entities":
                self.enable_entities = True
            elif stage.get("name") == "claims":
                self.enable_claims = True
            elif stage.get("name") == "relations":
                self.enable_relations = True
        
        # Set spaCy model if specified (via environment variable)
        # This must be done before importing spacy modules
        if self.spacy_model:
            os.environ["SPACY_MODEL"] = self.spacy_model
        
        # Set embedding model if specified (via environment variable)
        # This must be done before importing embedder modules
        if self.embed_model:
            os.environ["EMBEDDING_MODEL"] = self.embed_model
    
    def process_wikipedia_source(self, source_config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        """Process Wikipedia API source."""
        from okc_pipeline.stage_00_ingestion.doc_filters import keep_doc
        
        domain = source_config.get("domain", "en")
        seeds = source_config.get("seeds", [])
        max_docs = source_config.get("max_docs", 8000)
        max_hops = source_config.get("max_hops", 2)
        
        visited_titles: set[str] = set()
        seen_hashes: set[str] = set()
        q = deque((normalize_title(t), 0) for t in seeds)
        
        ingested = 0
        
        while q and ingested < max_docs:
            title, depth = q.popleft()
            if title in visited_titles:
                continue
            visited_titles.add(title)
            
            # Fetch text
            txt = fetch_wikipedia_article(title, domain)
            sleep_politely()
            if not txt:
                continue
            
            txt = clean_text(txt)
            if not keep_doc(txt, title=title.replace("_", " ")):
                continue
            
            h = hashlib.sha1(txt.encode("utf-8", errors="ignore")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            
            yield {
                "title": title.replace("_", " "),
                "url": f"https://{domain}.wikipedia.org/wiki/{title}",
                "text": txt,
                "lang": domain
            }
            
            ingested += 1
            
            # Expand links if we have depth budget
            if depth < max_hops:
                try:
                    links = get_wikipedia_links(title, domain)
                    sleep_politely()
                    random.shuffle(links)
                    for lt in links[:400]:  # cap fan-out
                        if lt not in visited_titles:
                            q.append((lt, depth + 1))
                except Exception:
                    pass
    
    def split_sentences_for_chunks(self, db, chunk_ids: List[int]):
        """Split chunks into sentences and store them in the database."""
        from okc_core.models import Chunk, Sentence
        from okc_pipeline.stage_01_sentences.sentence_splitter import split_chunk_into_sentences
        
        # Fetch all chunks at once for efficiency
        chunks = db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()
        
        for chunk in chunks:
            # Use the existing sentence splitter function
            sentence_data = split_chunk_into_sentences(chunk)
            
            # Insert sentences into database
            for order_idx, sent_dict in enumerate(sentence_data):
                sentence = Sentence(
                    chunk_id=chunk.id,
                    text=sent_dict["text"],
                    char_start=sent_dict["start"],
                    char_end=sent_dict["end"],
                    order_index=order_idx
                )
                db.add(sentence)
        
        db.flush()  # Flush to get IDs, but don't commit yet (let run() handle commits)
    
    def extract_entities_for_chunks(self, db, chunk_ids: List[int]):
        """Extract entities from sentences in chunks and store them in the database."""
        from okc_core.models import Chunk, Sentence
        from okc_pipeline.stage_02_entities.entity_extractor import extract_entities_for_sentences
        
        # Fetch all chunks at once for efficiency
        chunks = db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()
        
        # Collect all sentence IDs from these chunks
        sentence_ids = []
        for chunk in chunks:
            # Get sentences for this chunk
            chunk_sentences = db.query(Sentence).filter(Sentence.chunk_id == chunk.id).all()
            sentence_ids.extend([s.id for s in chunk_sentences])
        
        if sentence_ids:
            # Extract entities for all sentences
            stats = extract_entities_for_sentences(db, sentence_ids)
            print(f"[Pipeline] Extracted entities: {stats['total_mentions']} mentions from {stats['sentences_with_entities']} sentences")
        
        db.flush()  # Flush to get IDs, but don't commit yet (let run() handle commits)
    
    def detect_claim_sentences_for_chunks(self, db, chunk_ids: List[int]):
        """Detect claim sentences using Stage 3 heuristics."""
        from okc_core.models import Sentence
        from okc_pipeline.stage_03_claims.claim_detector import detect_claim_sentences
        
        if not chunk_ids:
            return
        
        sentence_id_rows = db.query(Sentence.id).filter(Sentence.chunk_id.in_(chunk_ids)).all()
        sentence_ids = [sid for (sid,) in sentence_id_rows]
        if not sentence_ids:
            print("[Pipeline] No sentences found for claim detection")
            return
        
        stats = detect_claim_sentences(db, sentence_ids)
        print(f"[Pipeline] Claim detection: processed {stats['sentences_processed']} sentences → {stats['claims_detected']} claims")
        db.flush()
    
    def extract_relations_for_chunks(self, db, chunk_ids: List[int]):
        """Extract relations from sentences in chunks."""
        from okc_core.models import Sentence
        from okc_pipeline.state_04_relations.relation_extractor import extract_relations_for_sentences
        
        if not chunk_ids:
            return
        
        sentence_id_rows = db.query(Sentence.id).filter(Sentence.chunk_id.in_(chunk_ids)).all()
        sentence_ids = [sid for (sid,) in sentence_id_rows]
        if not sentence_ids:
            print("[Pipeline] No sentences found for relation extraction")
            return
        
        stats = extract_relations_for_sentences(db, sentence_ids)
        print(f"[Pipeline] Relation extraction: processed {stats['sentences_processed']} sentences → {stats['relations_extracted']} relations")
        db.flush()
    
    def process_extracted_folder_source(self, source_config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        """Process extracted folder source."""
        from okc_pipeline.stage_00_ingestion.doc_filters import keep_doc
        
        folder_path = source_config.get("path", "extracted")
        max_docs = source_config.get("max_docs", 100)
        seen_hashes: set[str] = set()
        ingested = 0
        
        for path in iter_json_files(folder_path):
            if ingested >= max_docs:
                break
                
            for offset, doc in iter_json_lines(path):
                if ingested >= max_docs:
                    break
                    
                title = doc.get("title", "").strip()
                url = doc.get("url") or (f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" if title else None)
                text = doc.get("text", "").strip()
                
                if not title or not text:
                    continue
                
                if not keep_doc(text, title=title):
                    continue
                
                h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                
                yield {
                    "title": title,
                    "url": url,
                    "text": text,
                    "lang": "en"
                }
                
                ingested += 1
    
    def process_document(self, db, doc_data: Dict[str, Any]) -> Optional[int]:
        """Process a single document through all stages."""
        # Import here to ensure environment variable is set (model is cached on first import)
        from okc_pipeline.stage_00_embeddings.embedder import embed_texts
        from okc_pipeline.stage_00_embeddings.doc_embeddings import compute_doc_embedding
        from okc_pipeline.stage_00_ingestion.chunker import chunk_text
        from okc_pipeline.stage_00_ingestion.ingest_utils import detect_lang_optional
        from okc_api.crud import ingest_document, add_chunks_with_embeddings, update_document_embedding
        
        title = doc_data["title"]
        url = doc_data.get("url")
        text = doc_data["text"]
        lang = doc_data.get("lang") or detect_lang_optional(text) or "en"
        
        # Stage: Ingest document
        try:
            doc_vec = compute_doc_embedding(text) if text else None
        except ValueError:
            doc_vec = None
        
        doc_id, deduped = ingest_document(
            db, title, url, text, lang=lang, doc_embedding=doc_vec.tolist() if doc_vec is not None else None
        )
        
        if deduped:
            return None
        
        # Stage: Chunk
        target_tokens = self.chunk_params.get("target_tokens", 600)
        overlap = self.chunk_params.get("overlap", 80)
        chunks = chunk_text(text, target_tokens=target_tokens, overlap=overlap)
        
        # Stage: Embed chunks
        chunk_embeddings = embed_texts(chunks)
        
        # Add chunks with embeddings
        cids, doc_vec_updated = add_chunks_with_embeddings(
            db, doc_id, chunks, embeddings=chunk_embeddings
        )
        
        if doc_vec_updated is not None:
            update_document_embedding(db, doc_id, doc_vec_updated)
        
        # Stage: Sentence splitting (if enabled)
        if self.enable_sentences:
            self.split_sentences_for_chunks(db, cids)
        
        # Stage: Entity extraction (if enabled, requires sentences)
        if self.enable_entities:
            if not self.enable_sentences:
                print("[Pipeline] Warning: entities stage requires sentences stage, skipping entities")
            else:
                self.extract_entities_for_chunks(db, cids)
        
        # Stage: Claim sentence detection (requires sentences + entities)
        if self.enable_claims:
            if not self.enable_sentences:
                print("[Pipeline] Warning: claims stage requires sentences stage, skipping claims")
            elif not self.enable_entities:
                print("[Pipeline] Warning: claims stage requires entities stage, skipping claims")
            else:
                self.detect_claim_sentences_for_chunks(db, cids)
        
        # Stage: Relation extraction (requires sentences + entities)
        if self.enable_relations:
            if not self.enable_sentences:
                print("[Pipeline] Warning: relations stage requires sentences stage, skipping relations")
            elif not self.enable_entities:
                print("[Pipeline] Warning: relations stage requires entities stage, skipping relations")
            else:
                self.extract_relations_for_chunks(db, cids)
        
        return doc_id
    
    def process_existing_data(self, db):
        """Process existing data from database (when sourcing is skipped)."""
        from okc_core.models import Chunk, Sentence
        
        # Determine what we need to process based on enabled stages
        needs_chunks = self.enable_sentences or self.enable_entities or self.enable_claims or self.enable_relations
        needs_sentences = self.enable_entities or self.enable_claims or self.enable_relations
        
        if not needs_chunks and not needs_sentences:
            print("[Pipeline] No stages enabled that require existing data")
            return
        
        # Query existing chunks
        if needs_chunks:
            chunks = db.query(Chunk).all()
            if not chunks:
                print("[Pipeline] No existing chunks found in database")
                return
            
            chunk_ids = [c.id for c in chunks]
            print(f"[Pipeline] Found {len(chunk_ids)} existing chunks to process")
            
            # Process sentences if needed
            if self.enable_sentences:
                # Check which chunks don't have sentences yet
                chunks_without_sentences = [
                    c.id for c in chunks
                    if not db.query(Sentence).filter(Sentence.chunk_id == c.id).first()
                ]
                if chunks_without_sentences:
                    print(f"[Pipeline] Processing sentences for {len(chunks_without_sentences)} chunks")
                    self.split_sentences_for_chunks(db, chunks_without_sentences)
                    db.commit()
                else:
                    print("[Pipeline] All chunks already have sentences")
            
            sentence_ids_for_claims: list[int] = []
            
            # Process entities if needed
            if self.enable_entities:
                # Check which sentences don't have entities yet
                all_sentences = db.query(Sentence).filter(Sentence.chunk_id.in_(chunk_ids)).all()
                if not all_sentences:
                    print("[Pipeline] No sentences found. Run sentences stage first.")
                    return
                
                sentence_ids_for_claims = [s.id for s in all_sentences]
                
                # Check which sentences already have entity mentions
                from okc_core.models import EntityMention
                sentences_with_mentions_result = db.query(EntityMention.sentence_id).distinct().all()
                sentences_with_mentions = {s[0] for s in sentences_with_mentions_result}
                
                sentences_to_process = [
                    s.id for s in all_sentences if s.id not in sentences_with_mentions
                ]
                
                if sentences_to_process:
                    print(f"[Pipeline] Processing entities for {len(sentences_to_process)} sentences")
                    from okc_pipeline.stage_02_entities.entity_extractor import extract_entities_for_sentences
                    stats = extract_entities_for_sentences(db, sentences_to_process)
                    print(f"[Pipeline] Extracted entities: {stats['total_mentions']} mentions from {stats['sentences_with_entities']} sentences")
                    db.commit()
                else:
                    print("[Pipeline] All sentences already have entities")
            elif self.enable_claims:
                sentence_ids_for_claims = [
                    sid for (sid,) in db.query(Sentence.id).filter(Sentence.chunk_id.in_(chunk_ids)).all()
                ]
            
            if self.enable_claims:
                if not sentence_ids_for_claims:
                    print("[Pipeline] No sentences available for claim detection. Run sentences/entities first.")
                else:
                    from okc_pipeline.stage_03_claims.claim_detector import detect_claim_sentences
                    stats = detect_claim_sentences(db, sentence_ids_for_claims)
                    print(f"[Pipeline] Claim detection: processed {stats['sentences_processed']} sentences → {stats['claims_detected']} claims")
                    db.commit()
            
            if self.enable_relations:
                if not sentence_ids_for_claims:
                    print("[Pipeline] No sentences available for relation extraction. Run sentences/entities first.")
                else:
                    from okc_pipeline.state_04_relations.relation_extractor import extract_relations_for_sentences
                    stats = extract_relations_for_sentences(db, sentence_ids_for_claims)
                    print(f"[Pipeline] Relation extraction: processed {stats['sentences_processed']} sentences → {stats['relations_extracted']} relations")
                    db.commit()
        else:
            print("[Pipeline] No processing needed for existing data")
    
    def clear_database(self, db):
        """Clear all documents from database and all related data.
        
        Note: Entities and Relations are not directly linked to Documents via CASCADE,
        so they must be explicitly deleted. The deletion order matters due to foreign key constraints:
        1. RelationEvidence (references Relations and Sentences)
        2. Relations (they reference Entities)
        3. EntityMentions (they reference both Entities and Sentences - must be deleted before Entities)
        4. Entities (they're referenced by Relations and EntityMentions)
        5. Documents (cascades to Chunks → Sentences → ClaimSentences)
        """
        from okc_core.models import (
            Document, Entity, Relation, EntityMention, RelationEvidence
        )
        
        print("[Pipeline] Clearing database...")
        
        # Count before deletion for reporting
        doc_count = db.query(Document).count()
        entity_count = db.query(Entity).count()
        relation_count = db.query(Relation).count()
        mention_count = db.query(EntityMention).count()
        evidence_count = db.query(RelationEvidence).count()
        
        # Delete in order to respect foreign key constraints
        # Use flush() between deletions to ensure constraints are checked
        
        # 1. Delete RelationEvidence first (references Relations and Sentences)
        if evidence_count > 0:
            db.query(RelationEvidence).delete()
            db.flush()
            print(f"[Pipeline] Deleted {evidence_count} relation evidence records")
        
        # 2. Delete Relations (they reference Entities, must be deleted before Entities)
        if relation_count > 0:
            db.query(Relation).delete()
            db.flush()
            print(f"[Pipeline] Deleted {relation_count} relations")
        
        # 3. Delete Documents (cascades to Chunks → Sentences → EntityMentions, ClaimSentences)
        # This will delete EntityMentions via CASCADE, so we don't need to delete them explicitly
        if doc_count > 0:
            db.query(Document).delete()
            db.flush()
            print(f"[Pipeline] Deleted {doc_count} documents (cascaded to chunks, sentences, mentions, claims)")
        
        # 4. Delete Entities (now that EntityMentions are gone via Document CASCADE)
        if entity_count > 0:
            db.query(Entity).delete()
            db.flush()
            print(f"[Pipeline] Deleted {entity_count} entities")
        
        db.commit()
        
        # Verify deletion completed
        remaining_docs = db.query(Document).count()
        remaining_entities = db.query(Entity).count()
        remaining_relations = db.query(Relation).count()
        
        if doc_count == 0 and entity_count == 0 and relation_count == 0:
            print("[Pipeline] Database is already empty")
        else:
            print(f"[Pipeline] Database cleared: {doc_count} documents, {entity_count} entities, {relation_count} relations")
            if remaining_docs > 0 or remaining_entities > 0 or remaining_relations > 0:
                print(f"[Pipeline] WARNING: Some records remain - {remaining_docs} documents, {remaining_entities} entities, {remaining_relations} relations")
                raise RuntimeError("Database clearing incomplete - some records remain. This may cause constraint violations.")
    
    def run(self):
        """Run the pipeline."""
        from okc_core.db import SessionLocal
        
        db = SessionLocal()
        try:
            # Clear database if requested (must be done before any processing)
            if self.clear_db:
                self.clear_database(db)
                db.close()  # Close the old session
                db = SessionLocal()  # Get a fresh session after clearing
            
            if self.skip_sourcing:
                print("[Pipeline] Skipping sourcing - processing existing data from database")
                self.process_existing_data(db)
                db.commit()
                print("[Pipeline] Complete!")
            else:
                total_processed = 0
                total_accepted = 0
                
                for source_config in self.sources:
                    source_type = source_config.get("type")
                    
                    if source_type == "wikipedia_api":
                        print(f"[Pipeline] Processing Wikipedia API source: {source_config.get('seeds', [])}")
                        doc_iter = self.process_wikipedia_source(source_config)
                    elif source_type == "extracted_folder":
                        print(f"[Pipeline] Processing extracted folder source: {source_config.get('path', 'extracted')}")
                        doc_iter = self.process_extracted_folder_source(source_config)
                    else:
                        print(f"[Pipeline] Unknown source type: {source_type}, skipping...")
                        continue
                    
                    # Process documents
                    for doc_data in doc_iter:
                        doc_id = self.process_document(db, doc_data)
                        total_processed += 1
                        
                        if doc_id is not None:
                            total_accepted += 1
                        
                        if total_processed % 10 == 0:
                            print(f"[Pipeline] Processed: {total_processed}, Accepted: {total_accepted}")
                            db.commit()
                    
                    db.commit()
                
                print(f"[Pipeline] Complete! Processed: {total_processed}, Accepted: {total_accepted}")
        
        finally:
            db.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the OKC pipeline with YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages from config
  okc-pipeline pipeline.yaml
  
  # Skip entity extraction
  okc-pipeline pipeline.yaml --skip entities
  
  # Skip multiple stages
  okc-pipeline pipeline.yaml --skip entities --skip topics --skip claims
  
  # Process existing data (skip sourcing, chunk, embed, sentences)
  okc-pipeline pipeline.yaml --skip-sourcing --skip chunk --skip embed --skip sentences
  
  # Only run entity extraction on existing sentences
  okc-pipeline pipeline.yaml --skip-sourcing --skip chunk --skip embed --skip sentences --skip topics
  
  # Clear database and re-process everything (useful when changing chunking/entity extraction methods)
  okc-pipeline pipeline.yaml --clear-db
        """
    )
    parser.add_argument(
        "config",
        help="Path to pipeline YAML configuration file"
    )
    parser.add_argument(
        "--skip",
        action="append",
        dest="skip_stages",
        metavar="STAGE",
        help="Skip a stage (can be used multiple times). Valid stages: chunk, embed, sentences, entities, topics, claims, relations"
    )
    parser.add_argument(
        "--skip-sourcing",
        action="store_true",
        dest="skip_sourcing",
        help="Skip document sourcing and process existing data from database. Useful when data is already ingested."
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        dest="clear_db",
        help="Clear all documents from database before running (cascades to chunks, sentences, entities, etc.). "
             "Useful when changing chunking parameters or entity extraction methods to force re-processing."
    )
    
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    skip_stages = args.skip_stages if args.skip_stages else None
    runner = PipelineRunner(config_path, skip_stages=skip_stages, skip_sourcing=args.skip_sourcing, clear_db=args.clear_db)
    runner.run()


if __name__ == "__main__":
    main()

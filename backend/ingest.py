"""
Ingestion pipeline: chunk → embed → store in ChromaDB.
Run once (or re-run to refresh).  Idempotent — clears & rebuilds collection.

Usage:
    python -m backend.ingest
"""

import json
import re
import sys
import logging
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend import nim
from backend.config import (
    RAW_JSON_PATH, CHROMA_DIR, NIM_EMBED_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBED_BATCH_SIZE, EMBED_DIM,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bis.ingest")


# ── Chunking ────────────────────────────────────────────────────

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Recursive character-based splitter that approximates token count as
    chars / 4. Prefers splitting on double-newlines, then single, then spaces.
    """
    # Approximate character limits
    max_chars     = chunk_size * 4
    overlap_chars = overlap * 4

    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " ", ""]
    for sep in separators:
        if sep == "":
            # Hard split
            chunks = []
            start = 0
            while start < len(text):
                end = start + max_chars
                chunks.append(text[start:end].strip())
                start = end - overlap_chars
            return [c for c in chunks if c]

        parts = text.split(sep)
        if len(parts) < 2:
            continue

        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) > max_chars and current:
                chunks.append(current.strip())
                # Start new chunk with overlap: take last overlap_chars of current
                current = current[-overlap_chars:] + sep + part if current else part
            else:
                current = candidate
        if current.strip():
            chunks.append(current.strip())

        if all(len(c) <= max_chars for c in chunks):
            return [c for c in chunks if c]

    return [text.strip()]


# ── Embeddings via NVIDIA NIM ─────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed corpus chunks with nv-embedqa-e5-v5 as PASSAGES.

    The model is asymmetric — documents must use input_type="passage" while
    search queries use "query" (see retrieval.embed_query). Sending everything
    as one type silently degrades recall.
    """
    return nim.embed(texts, input_type="passage")


# ── ChromaDB ────────────────────────────────────────────────────

def get_collection(reset: bool = False):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection("bis_docs")
            log.info("Deleted existing ChromaDB collection.")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name="bis_docs",
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ── Main ingestion ───────────────────────────────────────────────

def run_ingestion(reset: bool = True):
    log.info("Loading documents from %s", RAW_JSON_PATH)
    with open(RAW_JSON_PATH, encoding="utf-8") as f:
        documents = json.load(f)

    # Filter out documents with useless content (< 100 chars)
    documents = [d for d in documents if len(d.get("raw_text", "")) >= 100]
    log.info("Using %d documents (filtered out short/empty ones)", len(documents))

    all_chunks:    list[str]  = []
    all_ids:       list[str]  = []
    all_metadatas: list[dict] = []

    for doc_idx, doc in enumerate(documents):
        text     = doc["raw_text"]
        url      = doc["source_url"]
        title    = doc["title"]
        category = doc["category"]
        stype    = doc["source_type"]

        chunks = split_text(text)
        log.info("  [%d/%d] '%s' → %d chunks", doc_idx + 1, len(documents), title, len(chunks))

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"doc{doc_idx:03d}_chunk{chunk_idx:04d}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "source_url":  url,
                "title":       title,
                "category":    category,
                "source_type": stype,
                "chunk_index": chunk_idx,
                "doc_index":   doc_idx,
            })

    from backend.config import NVIDIA_API_KEY
    all_embeddings = None
    if NVIDIA_API_KEY:
        log.info("Embedding with %s via NVIDIA NIM...", NIM_EMBED_MODEL)
        all_embeddings = []
        try:
            for i in range(0, len(all_chunks), EMBED_BATCH_SIZE):
                batch = all_chunks[i : i + EMBED_BATCH_SIZE]
                log.info("  Embedding batch %d-%d / %d", i + 1, i + len(batch), len(all_chunks))
                embs = embed_texts(batch)
                if len(embs) != len(batch):
                    raise RuntimeError(
                        f"Embedding count mismatch: sent {len(batch)}, got {len(embs)}"
                    )
                all_embeddings.extend(embs)
            if all_embeddings:
                dim = len(all_embeddings[0])
                log.info("Embedding dimension: %d", dim)
        except Exception as e:
            log.warning("Embedding skipped (%s) — storing documents for BM25 retrieval.", e)
            all_embeddings = None
    else:
        log.info("NVIDIA_API_KEY not set — storing documents directly into ChromaDB for BM25.")

    # Store in ChromaDB
    log.info("Storing in ChromaDB at %s ...", CHROMA_DIR)
    collection = get_collection(reset=reset)

    UPSERT_BATCH = 500
    for i in range(0, len(all_chunks), UPSERT_BATCH):
        upsert_kwargs = {
            "ids": all_ids[i : i + UPSERT_BATCH],
            "documents": all_chunks[i : i + UPSERT_BATCH],
            "metadatas": all_metadatas[i : i + UPSERT_BATCH],
        }
        if all_embeddings:
            upsert_kwargs["embeddings"] = all_embeddings[i : i + UPSERT_BATCH]
        collection.upsert(**upsert_kwargs)

    final_count = collection.count()
    log.info("Ingestion complete. ChromaDB collection has %d chunks.", final_count)
    return final_count


if __name__ == "__main__":
    run_ingestion(reset=True)

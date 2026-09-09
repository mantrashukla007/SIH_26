"""
Retrieval: hybrid BM25 + dense vector search with Reciprocal Rank Fusion.
Also provides structured QCO SQLite lookup.
"""

import re
import sqlite3
import logging
from typing import Optional

from rank_bm25 import BM25Okapi

from backend import nim
from backend.config import (
    QCO_DB_PATH, NIM_EMBED_MODEL,
    TOP_K_RETRIEVAL, TOP_K_FINAL, SIM_THRESHOLD,
    BM25_WEIGHT, DENSE_WEIGHT, CHROMA_DIR,
    QCO_STOPWORDS, QCO_MAX_ROWS,
)

log = logging.getLogger("bis.retrieval")

# ── Lazy-loaded singletons ───────────────────────────────────────
_collection      = None
_bm25_index      = None
_bm25_corpus     = None   # list of (doc_text, metadata)


def get_collection():
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            _collection = client.get_collection("bis_docs")
        except Exception:
            _collection = client.get_or_create_collection("bis_docs", metadata={"hnsw:space": "cosine"})
        
        # If collection is empty, attempt auto-ingestion from raw_documents.json
        if _collection.count() == 0:
            try:
                from backend.ingest import run_ingestion
                log.info("ChromaDB collection empty — running auto-ingestion...")
                run_ingestion(reset=False)
            except Exception as e:
                log.warning("Auto-ingestion failed: %s", e)
        log.info("ChromaDB collection loaded: %d chunks", _collection.count())
    return _collection


def _build_bm25():
    """Build BM25 index from all chunks in ChromaDB (with raw JSON fallback)."""
    global _bm25_index, _bm25_corpus
    if _bm25_index is not None:
        return
    log.info("Building BM25 index...")
    docs, metas, ids = [], [], []
    try:
        col = get_collection()
        result = col.get(include=["documents", "metadatas"])
        docs  = result.get("documents") or []
        metas = result.get("metadatas") or []
        ids   = result.get("ids") or []
    except Exception as exc:
        log.warning("Could not read ChromaDB for BM25 (%s) — using raw JSON fallback.", exc)

    if not docs:
        import json
        from backend.config import RAW_JSON_PATH
        if RAW_JSON_PATH.exists():
            try:
                with open(RAW_JSON_PATH, encoding="utf-8") as f:
                    raw_docs = json.load(f)
                for idx, doc in enumerate(raw_docs):
                    text = doc.get("raw_text", "").strip()
                    if len(text) >= 100:
                        ids.append(f"raw_doc_{idx}")
                        docs.append(text)
                        metas.append({
                            "source_url": doc.get("source_url", "https://www.bis.gov.in"),
                            "title": doc.get("title", "BIS Document"),
                            "category": doc.get("category", "general"),
                            "source_type": doc.get("source_type", "html"),
                        })
            except Exception as e:
                log.warning("Raw JSON fallback failed: %s", e)

    tokenized = [doc.lower().split() for doc in docs]
    if tokenized:
        _bm25_index  = BM25Okapi(tokenized)
    else:
        _bm25_index  = None
    _bm25_corpus = list(zip(ids, docs, metas))
    log.info("BM25 index built over %d chunks.", len(docs))


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string via NVIDIA NIM.

    input_type="query" is essential: nv-embedqa-e5-v5 is asymmetric, and the
    corpus was embedded as "passage".
    """
    return nim.embed_one(query, input_type="query")


# ── Structured QCO lookup ────────────────────────────────────────

def qco_keywords(query: str) -> list[str]:
    """Meaningful product keywords from a query (stopwords removed)."""
    words = re.findall(r"\b[\w/]{3,}\b", query.lower())
    return [w for w in dict.fromkeys(words) if w not in QCO_STOPWORDS]


def qco_structured_lookup(query: str) -> list[dict]:
    """
    Search the SQLite QCO table for product/standard matches.

    Rows are ranked by how many distinct query keywords they match (a product
    name hit counts double), so "standard for LED bulbs" surfaces the LED rows
    instead of an arbitrary alphabetical slice. Returns [] if nothing matches.
    """
    keywords = qco_keywords(query)
    if not keywords:
        return []

    # An explicit "IS 1417" style reference is the strongest possible signal
    is_refs = re.findall(r"\bis\s*(\d{2,5})", query.lower())

    conn = sqlite3.connect(QCO_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM qco_standards").fetchall()]
    except sqlite3.Error as e:
        log.warning("QCO DB query error: %s", e)
        return []
    finally:
        conn.close()

    scored: list[tuple[float, dict]] = []
    for row in rows:
        product = str(row.get("product_name", "")).lower()
        title   = str(row.get("standard_title", "")).lower()
        std_no  = str(row.get("is_standard_number", "")).lower()
        qco_ref = str(row.get("qco_reference", "")).lower()

        score = 0.0
        for kw in keywords[:8]:
            if kw in product:
                score += 2.0
            elif kw in title:
                score += 1.0
            elif kw in std_no or kw in qco_ref:
                score += 0.5
        for ref in is_refs:
            if ref in std_no:
                score += 5.0

        if score > 0:
            scored.append((score, row))

    if not scored:
        log.info("QCO structured lookup for '%s' → 0 rows", query[:60])
        return []

    scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("product_name", ""))))

    # Keep only rows close to the best match — avoids padding the prompt with
    # weak single-keyword hits that confuse the model.
    best   = scored[0][0]
    cutoff = max(1.0, best * 0.5)
    result = [row for score, row in scored if score >= cutoff][:QCO_MAX_ROWS]

    log.info(
        "QCO structured lookup for '%s' → %d rows (best score %.1f)",
        query[:60], len(result), best,
    )
    return result


# ── Reciprocal Rank Fusion ───────────────────────────────────────

def reciprocal_rank_fusion(
    dense_ids:  list[str],
    bm25_ids:   list[str],
    k:          int = 60,
) -> list[str]:
    """
    Combine two ranked lists via RRF.
    Returns merged list of IDs ordered by fused score (descending).
    """
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + DENSE_WEIGHT / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + BM25_WEIGHT / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ── Hybrid retrieval ─────────────────────────────────────────────

def hybrid_retrieve(query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """
    1. Dense vector search via ChromaDB (cosine similarity)
    2. BM25 keyword search over same corpus
    3. Fuse with RRF → return top_k chunks with metadata + scores

    If the NIM embeddings endpoint is unavailable, the dense leg is skipped and
    the answer is built from BM25 keyword retrieval alone — degraded, but far
    better than failing the whole request with a 500.
    """
    col = get_collection()
    _build_bm25()

    # ── Dense retrieval ──────────────────────────────────────────
    dense_ids:          list[str]   = []
    dense_docs:         list[str]   = []
    dense_metas:        list[dict]  = []
    dense_similarities: list[float] = []

    try:
        from backend.config import NVIDIA_API_KEY
        if NVIDIA_API_KEY:
            q_embedding = embed_query(query)
            dense_result = col.query(
                query_embeddings=[q_embedding],
                n_results=min(top_k, col.count()),
                include=["documents", "metadatas", "distances"],
            )
            dense_docs  = dense_result["documents"][0]
            dense_metas = dense_result["metadatas"][0]
            dense_ids   = dense_result["ids"][0]
            # Cosine similarity = 1 - distance (collection uses hnsw:space=cosine)
            dense_similarities = [1.0 - d for d in dense_result["distances"][0]]
    except Exception as e:
        log.warning(
            "Dense retrieval unavailable (%s: %s) — falling back to BM25 only.",
            type(e).__name__, e,
        )

    # ── BM25 retrieval ───────────────────────────────────────────
    if _bm25_index is not None:
        tokenized_query = query.lower().split()
        bm25_scores     = _bm25_index.get_scores(tokenized_query)

        # Top-k BM25 indices
        import numpy as np
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k].tolist()
        bm25_ids = [_bm25_corpus[i][0] for i in bm25_top_indices]
    else:
        bm25_scores = []
        bm25_top_indices = []
        bm25_ids = []

    # ── RRF fusion ───────────────────────────────────────────────
    fused_ids = reciprocal_rank_fusion(dense_ids, bm25_ids)[:top_k]

    # Build result dicts preserving similarity scores
    id_to_dense = {
        did: (doc, meta, sim)
        for did, doc, meta, sim in zip(dense_ids, dense_docs, dense_metas, dense_similarities)
    }
    id_to_bm25 = {
        _bm25_corpus[i][0]: (_bm25_corpus[i][1], _bm25_corpus[i][2])
        for i in bm25_top_indices
    }

    # Normalised BM25 score, used as a stand-in similarity for keyword-only hits
    bm25_max = float(bm25_scores[bm25_top_indices[0]]) if bm25_top_indices else 0.0
    bm25_sim = {
        _bm25_corpus[i][0]: (0.85 * (float(bm25_scores[i]) / bm25_max) if bm25_max > 0 else 0.0)
        for i in bm25_top_indices
    }

    results = []
    for chunk_id in fused_ids:
        if chunk_id in id_to_dense:
            doc_text, meta, sim = id_to_dense[chunk_id]
        elif chunk_id in id_to_bm25:
            doc_text, meta = id_to_bm25[chunk_id]
            # Without embeddings, BM25 top matches provide solid similarity signal
            sim = bm25_sim.get(chunk_id, 0.0) if not dense_ids else 0.0
        else:
            continue
        results.append({
            "chunk_id":   chunk_id,
            "text":       doc_text,
            "metadata":   meta,
            "similarity": round(sim, 4),
        })

    log.info("Hybrid retrieve: query='%s' → %d chunks (best sim=%.3f)",
             query[:60], len(results),
             max((r["similarity"] for r in results), default=0.0))
    return results


# ── Final selection ─────────────────────────────────────────────

def select_final(
    chunks: list[dict],
    top_k:  int = TOP_K_FINAL,
) -> list[dict]:
    """
    Trim fused candidates down to the passages actually sent to the LLM.

    A cross-encoder rerank would normally sit here, but NVIDIA retired its
    hosted reranking models (nv-rerankqa-*) in Aug 2026 and ships no hosted
    replacement, so the RRF order stands. Chunks with real dense similarity are
    preferred over BM25-only hits (similarity 0.0) so a keyword-only match
    cannot displace a semantically strong passage.
    """
    if len(chunks) <= top_k:
        return chunks

    best = sorted(
        enumerate(chunks),
        key=lambda pair: (-pair[1].get("similarity", 0.0), pair[0]),
    )[:top_k]
    keep = {i for i, _ in best}

    # Preserve the original fused ordering among the chunks we keep
    selected = [c for i, c in enumerate(chunks) if i in keep]
    log.info("Selected %d of %d candidates for the prompt", len(selected), len(chunks))
    return selected


# ── Confidence check ─────────────────────────────────────────────

def best_similarity(chunks: list[dict]) -> float:
    """
    Highest similarity among retrieved chunks.

    After RRF fusion the first chunk is not necessarily the most similar one
    (BM25-only hits carry a similarity of 0.0), so confidence must be taken
    across the whole set rather than from chunks[0].
    """
    if not chunks:
        return 0.0
    return max(c.get("similarity", 0.0) for c in chunks)


def above_threshold(chunks: list[dict]) -> bool:
    """Return True if the best retrieved chunk exceeds SIM_THRESHOLD."""
    return best_similarity(chunks) >= SIM_THRESHOLD

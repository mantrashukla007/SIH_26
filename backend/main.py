"""
FastAPI backend — BIS Intelligent Assistant (Phase 2)

Endpoints:
  POST /chat    — main Q&A endpoint
  GET  /health  — health + index stats
  POST /ingest  — trigger re-ingestion (admin)
"""

import logging
import sys
import json
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import TOP_K_FINAL, TOP_K_RETRIEVAL, SIM_THRESHOLD
from backend.classifier import classify_query
from backend.followup import resolve_query
from backend.retrieval import (
    hybrid_retrieve, qco_structured_lookup, above_threshold, best_similarity,
    select_final, get_collection,
)
from backend.generation import generate_answer, build_context_block, build_qco_context, NOT_ENOUGH_INFO
from backend import nim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("bis.api")

app = FastAPI(
    title="BIS Intelligent Assistant API",
    description=(
        "RAG-based Q&A for Bureau of Indian Standards — "
        "zero-hallucination answers grounded in official BIS sources."
    ),
    version="1.0.0",
)


@app.on_event("startup")
async def _pre_warm():
    """Pre-build the BM25 index and warm the ChromaDB connection at startup.

    Without this, the very first query pays a cold-start penalty of several
    seconds while the BM25 index is built from all ChromaDB chunks. Moving it
    to startup means users never feel that delay.
    """
    import asyncio
    from backend.retrieval import get_collection
    try:
        loop = asyncio.get_event_loop()
        # Run blocking I/O in a thread so startup doesn't block the event loop
        # We fire-and-forget here so Uvicorn can finish startup immediately.
        loop.run_in_executor(None, _warm_retrieval)
        log.info("Pre-warm initiated in background...")
    except Exception as exc:
        log.warning("Pre-warm failed to start: %s", exc)


def _warm_retrieval():
    from backend.retrieval import get_collection, _build_bm25
    get_collection()   # opens ChromaDB
    _build_bm25()      # builds BM25 from the corpus

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "BIS Intelligent Assistant API"}

# ── Request / Response models ────────────────────────────────────

class HistoryMessage(BaseModel):
    role:    Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class ChatRequest(BaseModel):
    query:   str                          = Field(..., min_length=3, max_length=1000)
    history: list[HistoryMessage]         = Field(default_factory=list)
    use_nim: bool                         = True   # reserved; NIM is the only provider


class SourceInfo(BaseModel):
    title:      str
    url:        str
    category:   str
    similarity: float


class ChatResponse(BaseModel):
    answer:          str
    sources:         list[SourceInfo]
    route:           str    # which pipeline was used
    low_confidence:  bool
    stripped_is:     list[str]   # IS numbers removed by guardrail
    used_nim:        bool
    confidence:      float       # top chunk similarity score
    carried_terms:   list[str] = []    # topic terms pulled from earlier turns


RETRIEVAL_DOWN_RESPONSE = (
    "I couldn't search my document index just now, so I won't guess an answer.\n\n"
    "If you are running this locally, check that the index has been built "
    "(`python -m backend.ingest`) and that `NVIDIA_API_KEY` is set in `.env`. "
    "Meanwhile, official information is available at "
    "[bis.gov.in](https://www.bis.gov.in) or BIS CARE **1800-11-4000**."
)

OUT_OF_SCOPE_RESPONSE = (
    "I'm designed to answer questions about Bureau of Indian Standards (BIS) — "
    "including product certification, hallmarking, Indian Standards, and quality control orders.\n\n"
    "Your question appears to be outside my scope. Please visit "
    "[bis.gov.in](https://www.bis.gov.in) for official information, "
    "or call the BIS CARE helpline at **1800-11-4000**."
)


# ── Chat endpoint ────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    query = req.query.strip()
    log.info("QUERY: %s", query)

    history_msgs = [{"role": m.role, "content": m.content} for m in req.history]

    # 1. Resolve follow-ups. "Is it mandatory?" is meaningless to a retriever on
    #    its own, so topic terms from earlier turns are folded into the SEARCH
    #    query only. `query` stays exactly what the user typed and is what the
    #    LLM is asked to answer.
    search_query, carried_terms = resolve_query(query, history_msgs)

    # 2. Classify the resolved text — a bare "is it mandatory?" carries no
    #    routing signal, while "is it mandatory? led bulbs IS 16102" routes to
    #    the structured lookup where it belongs.
    route = classify_query(search_query)

    if route == "out_of_scope":
        return ChatResponse(
            answer         = OUT_OF_SCOPE_RESPONSE,
            sources        = [],
            route          = route,
            low_confidence = False,
            stripped_is    = [],
            used_nim       = False,
            confidence     = 0.0,
            carried_terms  = carried_terms,
        )

    # 3. Structured QCO lookup. Run it for every in-scope route: questions like
    #    "is a helmet mandatory?" often classify as FAQ yet are answered best
    #    from the deterministic table. The lookup is keyword-scored, so it
    #    returns nothing when the query names no product.
    qco_rows = qco_structured_lookup(search_query)
    if qco_rows:
        log.info("Structured match: %d QCO rows", len(qco_rows))

    # 4. Hybrid RAG retrieval. Never let an infrastructure hiccup (NIM
    #    unreachable, missing index) turn into an opaque 500 for the user.
    try:
        raw_chunks = hybrid_retrieve(search_query, top_k=TOP_K_RETRIEVAL)
    except Exception as e:
        log.error("Retrieval failed (%s): %s", type(e).__name__, e)
        raw_chunks = []
        if not qco_rows:
            return ChatResponse(
                answer         = RETRIEVAL_DOWN_RESPONSE,
                sources        = [],
                route          = route,
                low_confidence = True,
                stripped_is    = [],
                used_nim       = False,
                confidence     = 0.0,
                carried_terms  = carried_terms,
            )

    # Filter by category for more targeted retrieval, then trim to the final
    # prompt budget.
    if route == "procedure_rag":
        preferred = [c for c in raw_chunks if c["metadata"].get("category") in
                     ("certification", "qco", "hallmarking")]
        candidates = preferred or raw_chunks
    elif route == "faq_rag":
        preferred = [c for c in raw_chunks if c["metadata"].get("category") in
                     ("faq", "consumer", "hallmarking")]
        candidates = preferred or raw_chunks
    else:
        candidates = raw_chunks

    chunks = select_final(candidates, top_k=TOP_K_FINAL)

    # Confidence must come from the best chunk in the set: after RRF fusion the
    # first chunk can be a BM25-only hit whose similarity is 0.0.
    confidence    = best_similarity(chunks)
    low_conf      = not above_threshold(chunks) and not qco_rows

    # If both structured and vector retrieval found nothing useful → decline
    if not qco_rows and not chunks:
        return ChatResponse(
            answer         = NOT_ENOUGH_INFO,
            sources        = [],
            route          = route,
            low_confidence = True,
            stripped_is    = [],
            used_nim       = False,
            confidence     = 0.0,
            carried_terms  = carried_terms,
        )

    # 5. Flag if falling back to RAG for a structured question
    if route == "structured_lookup" and not qco_rows:
        # No deterministic match — augment context with disclaimer
        log.warning("Structured lookup returned 0 rows — using RAG fallback")
        # Prepend note so LLM includes it in answer
        chunks_with_note = chunks
        rag_fallback_note = (
            "(Note: No confirmed QCO match found in the structured database. "
            "The following is based on general guidance documents — "
            "please verify at bis.gov.in for the authoritative QCO status.)"
        )
        if chunks_with_note:
            chunks_with_note[0]["text"] = rag_fallback_note + "\n\n" + chunks_with_note[0]["text"]
    else:
        chunks_with_note = chunks

    # 6. Generate answer. The resolved query goes along only when it differs, so
    #    the model knows which topic the retrieved passages were chosen for.
    result = generate_answer(
        query          = query,
        chunks         = chunks_with_note,
        qco_rows       = qco_rows,
        history        = history_msgs,
        low_confidence = low_conf,
        route          = route,
        resolved_query = search_query if carried_terms else None,
    )

    return ChatResponse(
        answer         = result["answer"],
        sources        = [SourceInfo(**s) for s in result["sources"]],
        route          = route,
        low_confidence = result["low_confidence"],
        stripped_is    = result["stripped_is"],
        used_nim       = result["used_nim"],
        confidence     = confidence,
        carried_terms  = carried_terms,
    )


# ── Streaming chat endpoint ─────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming version of /chat using Server-Sent Events (SSE).

    Event types sent to the client:
      data: {"type": "meta",  ...route/confidence/sources metadata}
      data: {"type": "token", "text": "..."}
      data: {"type": "done"}
      data: {"type": "error", "message": "..."}
    """
    from backend.generation import (
        build_context_block, build_qco_context,
        strip_thinking, strip_hallucinated_is_numbers,
        LOW_CONFIDENCE_NOTE, SYSTEM_PROMPT, ROUTE_INSTRUCTIONS,
    )
    from backend.config import THINKING_TOGGLE, MAX_HISTORY_TURNS

    query = req.query.strip()
    log.info("STREAM QUERY: %s", query)
    history_msgs = [{"role": m.role, "content": m.content} for m in req.history]

    search_query, carried_terms = resolve_query(query, history_msgs)
    route = classify_query(search_query)

    async def event_generator():
        # ── Out-of-scope fast-path ─────────────────────────────
        if route == "out_of_scope":
            yield _sse({"type": "meta", "route": route, "confidence": 0.0,
                        "low_confidence": False, "sources": [],
                        "stripped_is": [], "used_nim": False,
                        "carried_terms": carried_terms})
            yield _sse({"type": "token", "text": OUT_OF_SCOPE_RESPONSE})
            yield _sse({"type": "done"})
            return

        # ── Structured + RAG retrieval ─────────────────────────
        qco_rows = qco_structured_lookup(search_query)
        try:
            raw_chunks = hybrid_retrieve(search_query, top_k=TOP_K_RETRIEVAL)
        except Exception as exc:
            log.error("Retrieval failed: %s", exc)
            raw_chunks = []
            if not qco_rows:
                yield _sse({"type": "meta", "route": route, "confidence": 0.0,
                            "low_confidence": True, "sources": [],
                            "stripped_is": [], "used_nim": False,
                            "carried_terms": carried_terms})
                yield _sse({"type": "token", "text": RETRIEVAL_DOWN_RESPONSE})
                yield _sse({"type": "done"})
                return

        # Filter by route category
        if route == "procedure_rag":
            preferred = [c for c in raw_chunks
                         if c["metadata"].get("category") in ("certification", "qco", "hallmarking")]
            candidates = preferred or raw_chunks
        elif route == "faq_rag":
            preferred = [c for c in raw_chunks
                         if c["metadata"].get("category") in ("faq", "consumer", "hallmarking")]
            candidates = preferred or raw_chunks
        else:
            candidates = raw_chunks

        chunks      = select_final(candidates, top_k=TOP_K_FINAL)
        confidence  = best_similarity(chunks)
        low_conf    = not above_threshold(chunks) and not qco_rows

        if not qco_rows and not chunks:
            yield _sse({"type": "meta", "route": route, "confidence": 0.0,
                        "low_confidence": True, "sources": [],
                        "stripped_is": [], "used_nim": False,
                        "carried_terms": carried_terms})
            yield _sse({"type": "token", "text": NOT_ENOUGH_INFO})
            yield _sse({"type": "done"})
            return

        # RAG fallback note
        chunks_with_note = chunks
        if route == "structured_lookup" and not qco_rows:
            note = ("(Note: No confirmed QCO match found. "
                    "The following is based on general guidance — verify at bis.gov.in.)")
            if chunks_with_note:
                chunks_with_note[0]["text"] = note + "\n\n" + chunks_with_note[0]["text"]

        # ── Build prompt ───────────────────────────────────────
        context_parts = []
        if qco_rows:
            context_parts.append(build_qco_context(qco_rows))
        if chunks_with_note:
            context_parts.append(build_context_block(chunks_with_note))
        context_block = "\n\n".join(context_parts)

        route_note    = ROUTE_INSTRUCTIONS.get(route, "")
        resolved_note = (f"RESOLVED TOPIC: {search_query}\n\n" if carried_terms else "")

        user_content = (
            f"RETRIEVED CONTEXT (the only material you may use):\n{context_block}\n\n"
            f"USER QUESTION: {query}\n\n"
            + resolved_note
            + (f"ROUTING NOTE: {route_note}\n\n" if route_note else "")
            + "Answer strictly from the context above. "
              "Every IS number must appear in it verbatim. "
              "Finish with `Source: <document title(s)>`. "
              "If the context does not cover the question, use the 'not enough verified information' reply."
        )

        messages = (
            [{"role": "system", "content": f"{THINKING_TOGGLE}\n\n{SYSTEM_PROMPT}"}]
            + [{"role": m["role"], "content": m["content"]}
               for m in (history_msgs or [])
               if m.get("role") in ("user", "assistant") and m.get("content")
              ][-(MAX_HISTORY_TURNS * 2):]
            + [{"role": "user", "content": user_content}]
        )

        # ── Build sources list (sent as metadata before tokens) ─
        seen_urls: set = set()
        sources = []
        for row in (qco_rows or []):
            url = str(row.get("source_url", "") or "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({"title": str(row.get("qco_reference") or "BIS QCO"),
                                 "url": url, "category": "qco", "similarity": 1.0})
        for chunk in chunks:
            url = chunk["metadata"].get("source_url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                sources.append({"title": chunk["metadata"].get("title", ""),
                                 "url": url,
                                 "category": chunk["metadata"].get("category", ""),
                                 "similarity": chunk["similarity"]})

        # ── Send metadata FIRST so the UI can render sources immediately
        yield _sse({"type": "meta", "route": route,
                    "confidence": round(confidence, 4),
                    "low_confidence": low_conf,
                    "sources": sources,
                    "stripped_is": [], "used_nim": True,
                    "carried_terms": carried_terms})

        # ── Stream tokens ─────────────────────────────────────
        full_text = ""
        try:
            for delta in nim.chat_stream(messages):
                full_text += delta
                # Strip <think> tags on the fly
                clean = delta
                if "<think>" in full_text or "</think>" in full_text:
                    clean = ""
                yield _sse({"type": "token", "text": clean})
        except Exception as exc:
            log.error("NIM stream error: %s", exc)
            yield _sse({"type": "error", "message": str(exc)})
            return

        # ── Post-process: guardrail (for stripped_is final update) ─
        from backend.generation import strip_hallucinated_is_numbers, strip_thinking
        full_text = strip_thinking(full_text)
        _, stripped_is = strip_hallucinated_is_numbers(full_text, chunks, qco_rows)

        if low_conf:
            yield _sse({"type": "token",
                        "text": "\n\n⚠️ *Low confidence*: Please verify at "
                                "[bis.gov.in](https://www.bis.gov.in) or call 1800-11-4000."})

        # Send final stripped_is update
        if stripped_is:
            yield _sse({"type": "stripped_is", "items": stripped_is})

        yield _sse({"type": "done"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── Health endpoint ──────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        col   = get_collection()
        count = col.count()
        status = "ok"
    except Exception as e:
        count  = 0
        status = f"chroma_error: {e}"

    from backend.config import (
        NVIDIA_API_KEY, GROQ_API_KEY, QCO_DB_PATH,
        GROQ_CHAT_MODEL, NIM_EMBED_MODEL, EMBED_DIM,
    )
    import sqlite3
    try:
        conn      = sqlite3.connect(QCO_DB_PATH)
        qco_count = conn.execute("SELECT COUNT(*) FROM qco_standards").fetchone()[0]
        conn.close()
    except Exception:
        qco_count = 0

    return {
        "status":           status,
        "chroma_chunks":    count,
        "qco_table_rows":   qco_count,
        "provider":         "Groq + NVIDIA NIM",
        "llm_model":        GROQ_CHAT_MODEL,
        "embed_model":      NIM_EMBED_MODEL,
        "embed_dim":        EMBED_DIM,
        "rerank_model":     None,
        "nvidia_key_set":   bool(NVIDIA_API_KEY),
        "groq_key_set":     bool(GROQ_API_KEY),
    }


# ── Ingest endpoint (admin) ──────────────────────────────────────

@app.post("/ingest")
async def trigger_ingest():
    """Re-run ingestion pipeline. Clears and rebuilds ChromaDB collection."""
    try:
        from backend.ingest import run_ingestion
        count = run_ingestion(reset=True)
        return {"status": "ok", "chunks_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

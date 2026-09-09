"""
rebuild_qco.py — Clean + rebuild the QCO structured database.

What this does:
  1. Deletes rows whose product_name is purely numeric (the garbage rows
     from the v1 parser that extracted serial numbers instead of names).
  2. Re-inserts the 65 curated seed rows from bis_supplement.py.
  3. Re-extracts rows from already-downloaded PDFs using the v2 parser.
  4. Prints a before/after summary.

Run this ONCE after upgrading to the v2 parser, without re-scraping the web.
"""

import sys
import sqlite3
import re
import logging
from pathlib import Path
import pymupdf as fitz

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rebuild_qco")

QCO_DB  = Path("data/qco_structured.db")
PDF_DIR = Path("data/pdfs")


# ── Import seed data and parser from bis_supplement ─────────────
sys.path.insert(0, str(Path(__file__).parent))
from scraper.bis_supplement import (   # noqa: E402
    SEED_QCO_ROWS, parse_qco_rows, upsert_qco_sqlite,
)


# ── Regexes for detecting garbage product names ──────────────────
_NUMERIC_ONLY   = re.compile(r"^[\d\s.,;:()/-]{1,12}$")
_HEADER_WORDS   = re.compile(
    r"^(?:s\.?\s*no\.?|sr\.?|serial|sl\.?|product|standard|title|"
    r"is\s*no\.?|is\s*number|scheme|status|mandatory|compulsory|"
    r"category|item|particulars|description)\s*$",
    re.I,
)


def is_garbage_name(name: str) -> bool:
    if not name or len(name.strip()) < 4:
        return True
    if _NUMERIC_ONLY.match(name.strip()):
        return True
    if _HEADER_WORDS.match(name.strip()):
        return True
    return False


def main():
    QCO_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(QCO_DB)
    conn.row_factory = sqlite3.Row

    # ── Ensure table exists ─────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS qco_standards (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name           TEXT,
            is_standard_number     TEXT,
            standard_title         TEXT,
            mandatory_or_voluntary TEXT,
            scheme_type            TEXT,
            qco_reference          TEXT,
            source_url             TEXT,
            penalty_clause         TEXT,
            UNIQUE(product_name, is_standard_number)
        )
    """)
    conn.commit()

    # If table is empty and CSV exists, load from CSV
    count_now = conn.execute("SELECT COUNT(*) FROM qco_standards").fetchone()[0]
    csv_path = Path("data/qco_structured.csv")
    if count_now == 0 and csv_path.exists():
        log.info("Loading initial records from %s ...", csv_path)
        import csv
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        upsert_qco_sqlite(csv_rows)
        log.info("Loaded %d rows from CSV into SQLite.", len(csv_rows))

    before = conn.execute("SELECT COUNT(*) FROM qco_standards").fetchone()[0]
    log.info("=== Current total rows in qco_standards: %d ===", before)

    # ── 1. Find and delete garbage rows ────────────────────────
    all_rows = conn.execute(
        "SELECT id, product_name FROM qco_standards"
    ).fetchall()

    garbage_ids = [r["id"] for r in all_rows if is_garbage_name(r["product_name"])]
    log.info("Found %d garbage rows (numeric / header product names) — deleting...",
             len(garbage_ids))

    if garbage_ids:
        placeholders = ",".join("?" * len(garbage_ids))
        conn.execute(f"DELETE FROM qco_standards WHERE id IN ({placeholders})",
                     garbage_ids)
        conn.commit()

    after_clean = conn.execute("SELECT COUNT(*) FROM qco_standards").fetchone()[0]
    log.info("After cleanup: %d rows remain.", after_clean)
    conn.close()

    # ── 2. Re-insert seed rows ──────────────────────────────────
    log.info("Upserting %d curated seed rows...", len(SEED_QCO_ROWS))
    upsert_qco_sqlite(SEED_QCO_ROWS)

    # ── 3. Re-extract from downloaded PDFs ─────────────────────
    pdfs = sorted(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []
    log.info("Re-extracting from %d downloaded PDFs with v2 parser...", len(pdfs))

    all_pdf_rows = []
    for pdf_path in pdfs:
        log.info("  Processing: %s", pdf_path.name)
        try:
            pdf_bytes = pdf_path.read_bytes()
            pages = []
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for i, page in enumerate(doc, 1):
                    t = page.get_text("text").strip()
                    if t:
                        pages.append(f"[Page {i}]\n{t}")
            text = "\n\n".join(pages)

            title = pdf_path.stem.replace("-", " ").replace("_", " ").title()
            url   = f"local://data/pdfs/{pdf_path.name}"
            rows  = parse_qco_rows(text, url, title)
            if rows:
                upsert_qco_sqlite(rows)
                all_pdf_rows.extend(rows)
                log.info("    Added %d rows from '%s'", len(rows), title)
        except Exception as exc:
            log.warning("    Failed to process %s: %s", pdf_path.name, exc)

    # ── 4. Final summary ────────────────────────────────────────
    conn = sqlite3.connect(QCO_DB)
    conn.row_factory = sqlite3.Row
    final = conn.execute("SELECT COUNT(*) FROM qco_standards").fetchone()[0]

    print("\n" + "=" * 70)
    print("REBUILD COMPLETE")
    print("=" * 70)
    print(f"  Rows BEFORE cleanup  : {before}")
    print(f"  Garbage rows deleted : {len(garbage_ids)}")
    print(f"  Seed rows upserted   : {len(SEED_QCO_ROWS)}")
    print(f"  PDF-extracted rows   : {len(all_pdf_rows)}")
    print(f"  TOTAL rows NOW       : {final}")
    print()
    print("Sample rows after rebuild (product_name should be real names now):")
    print(f"  {'IS Standard':<25} | {'Product Name':<55} | Status")
    print("  " + "-" * 95)
    for row in conn.execute(
        "SELECT is_standard_number, product_name, mandatory_or_voluntary "
        "FROM qco_standards ORDER BY product_name LIMIT 25"
    ):
        std     = (row["is_standard_number"] or "")[:25]
        product = (row["product_name"]       or "")[:55]
        status  = row["mandatory_or_voluntary"] or ""
        print(f"  {std:<25} | {product:<55} | {status}")

    conn.close()
    print("=" * 70)


if __name__ == "__main__":
    main()

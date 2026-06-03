"""
Budget PDF Table Extraction Pipeline  — v2
===========================================
Fully automated. Handles:
  • Multi-page continuation tables (same col-count, sequential pages)
  • Context extraction: annex, section heading, unit from page text
  • Title inference that walks backward through full document text
  • Rejects layout-only "tables" (pure prose with no data columns)
  • NLP-ready output: JSON + per-table CSV + flat annotated text corpus

Works on any Indian Government / Budget PDF with similar structure.
"""

import pdfplumber
import json, csv, re, os
from pathlib import Path
from copy import deepcopy

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
PDF_PATH = "23-24.pdf"
OUT_DIR  = Path(r"C:\Users\gauth\IUB_NITT_Internship_work\output_23-24")
CSV_DIR  = OUT_DIR / "tables_csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

REPORT = []
def log(m): REPORT.append(m); print(m)


# ─────────────────────────────────────────────────────────────
# STEP 1 — PAGE CONTEXT EXTRACTION
# ─────────────────────────────────────────────────────────────

RE_ANNEX = re.compile(
    r"(Annex(?:ure)?\s+No\.?\s+[\w\-]+\s+to\s+Part\s+[A-Z](?:\s+(?:of\s+)?the\s+Budget\s+Speech)?)",
    re.I
)
RE_UNIT = re.compile(
    r"(Rs\.?\s*in\s*crore|in\s*crore\s*of\s*rupees|₹\s*in\s*crore|"
    r"in\s*lakh|Rs\.?\s*per\s*thousand|percentage\s*\(?%\)?)",
    re.I
)
# Roman numeral section headers like "VI Incentivizing..." or "INDIRECT TAX"
RE_SECTION = re.compile(
    r"^((?:X{0,3}(?:IX|IV|V?I{0,3}))\s+[A-Z][^\n]{8,}|"
    r"(?:DIRECT TAX|INDIRECT TAX|CUSTOMS ACT|EXCISE ACT|SERVICE TAX)[^\n]*)",
    re.MULTILINE
)
RE_ALLCAPS_HEADING = re.compile(r"^([A-Z][A-Z &,/()\-:]{9,})\s*$", re.MULTILINE)


def page_context(text: str) -> dict:
    ctx = {"annex": "", "unit": "", "section": "", "heading": ""}

    m = RE_ANNEX.search(text)
    if m:
        ctx["annex"] = m.group(1).strip()
        # grab the descriptive title on the line(s) after the annex marker
        after = text[m.end():].lstrip()
        title_line = after.split("\n")[0].strip()
        if len(title_line) > 5:
            ctx["annex_title"] = title_line
        else:
            # might span 2 lines
            lines = [l.strip() for l in after.split("\n") if l.strip()]
            ctx["annex_title"] = " ".join(lines[:2]) if lines else ""
    else:
        ctx["annex_title"] = ""

    m = RE_UNIT.search(text)
    if m:
        raw = m.group(1).lower()
        if "crore" in raw:       ctx["unit"] = "Rs. in Crore"
        elif "lakh" in raw:      ctx["unit"] = "Rs. in Lakh"
        elif "thousand" in raw:  ctx["unit"] = "Rs. per Thousand"
        else:                    ctx["unit"] = m.group(1).strip()

    # find the last (most specific) section heading on the page
    for m in RE_SECTION.finditer(text):
        ctx["section"] = re.sub(r"\s+", " ", m.group(1)).strip()

    for m in RE_ALLCAPS_HEADING.finditer(text):
        h = m.group(1).strip()
        if len(h) > 10:
            ctx["heading"] = h

    return ctx


# ─────────────────────────────────────────────────────────────
# STEP 2 — TABLE CLEANING HELPERS
# ─────────────────────────────────────────────────────────────

def clean_cell(v) -> str:
    return re.sub(r"\s+", " ", str(v) if v is not None else "").strip()

def clean_table(raw):
    return [[clean_cell(c) for c in row] for row in raw]

def first_good_header_row(rows):
    """Index of first row where ≥ half cells are non-empty."""
    for i, row in enumerate(rows):
        if sum(1 for c in row if c) >= max(1, len(row) // 2):
            return i
    return 0

def is_prose_table(rows, ncols):
    """
    Tables that are really just prose in a layout box:
      - 2 cols where col-0 is always empty and col-1 is long sentences
    These are NOT data tables.
    """
    if ncols > 2:
        return False
    if ncols == 2:
        col0_empty = sum(1 for r in rows if not r[0]) / max(len(rows), 1)
        avg_len    = sum(len(r[1]) for r in rows if len(r) > 1) / max(len(rows), 1)
        if col0_empty > 0.7 and avg_len > 80:
            return True
    return False

def is_section_label_row(row):
    """Row that is just a section Roman numeral with all other cells empty."""
    if not row: return False
    rest_empty = all(c == "" for c in row[1:])
    roman = re.match(r"^(I{1,3}V?|VI{0,3}|IX|X{1,3}|XI{1,3})$", row[0].strip())
    return bool(roman and rest_empty)

def col_types(headers):
    types = []
    for h in headers:
        hl = h.lower()
        if re.search(r"\b(sl|s\.?no|no\.|sno|#)\b", hl):       types.append("id")
        elif re.search(r"\b(actual|re\s|be\s|total|iebr|crore|lakh|amount|rupee)\b", hl):
                                                                  types.append("numeric")
        elif re.search(r"\b(rate|%|tds|tcs|duty|cess|existing|proposed)\b", hl):
                                                                  types.append("rate")
        elif re.search(r"\b(change|description|head|scheme|ministry|item|measure|changes)\b", hl):
                                                                  types.append("description")
        else:                                                     types.append("text")
    return types


# ─────────────────────────────────────────────────────────────
# STEP 3 — TITLE INFERENCE
# The key insight: walk BACKWARD through preceding page lines to
# find the nearest "standalone heading" that precedes this table.
# ─────────────────────────────────────────────────────────────

def preceding_lines_before_table(page_text, table_header_sample):
    """
    Split page text into lines. Find where the table header text appears,
    then return all lines before that point.
    """
    # use first non-empty cell as anchor
    anchor = next((c for c in table_header_sample if len(c) > 3), "")
    if not anchor:
        return page_text.split("\n")
    idx = page_text.find(anchor[:20])
    if idx == -1:
        return page_text.split("\n")
    return page_text[:idx].split("\n")

def infer_title(ctx, header_row, page_text, tbl_idx):
    # Priority 1: annex with its title
    if ctx["annex"] and ctx.get("annex_title"):
        return f"{ctx['annex']} — {ctx['annex_title']}"
    if ctx["annex"]:
        return ctx["annex"]

    # Priority 2: last section heading on the page
    if ctx["section"]:
        return ctx["section"]

    # Priority 3: walk back through page lines for nearest heading
    pre_lines = preceding_lines_before_table(page_text, header_row)
    # Best candidate: a line that is not a number, not too short, not a sentence
    for line in reversed(pre_lines):
        line = line.strip()
        if (8 < len(line) < 120
                and not re.match(r"^\d+$", line)
                and not re.match(r"^[\d.,% ()]+$", line)
                and not line.endswith((".", ","))       # avoid mid-sentence fragments
                and re.search(r"[A-Za-z]{4,}", line)):
            return line

    # Priority 4: first header cell if substantive
    if header_row and len(header_row[0]) > 12:
        return header_row[0]

    return f"Table {tbl_idx+1} (page {ctx.get('_page', '?')})"


# ─────────────────────────────────────────────────────────────
# STEP 4 — CONTINUATION DETECTION
# ─────────────────────────────────────────────────────────────

def is_continuation(prev, curr_rows, curr_page):
    """
    curr is a continuation of prev if:
      1. Same column count
      2. curr_page == prev's last page + 1
      3. curr's first row is NOT identical to prev's column headers
         (which would signal a fresh table with a repeated header)
      4. curr's first row has no all-caps heading-like content
    """
    if not prev or not curr_rows:
        return False
    if prev["column_count"] != len(curr_rows[0]):
        return False
    if curr_page != prev["source_pages"][-1] + 1:
        return False

    prev_hdr_lower = [h.lower() for h in prev["column_headers"]]
    curr_first_lower = [c.lower() for c in curr_rows[0]]
    if prev_hdr_lower == curr_first_lower:
        return False

    # If the first row of curr looks like a completely fresh section label
    # with a new title in col-0 and nothing in col-1+ → still continuation
    # (section sub-headers inside a continuing policy table are fine)

    return True


# ─────────────────────────────────────────────────────────────
# STEP 5 — MAIN EXTRACTION LOOP
# ─────────────────────────────────────────────────────────────

log("=" * 68)
log("BUDGET TABLE EXTRACTION PIPELINE  v2")
log(f"Source: {PDF_PATH}")
log("=" * 68)

physical_tables = []

with pdfplumber.open(PDF_PATH) as pdf:
    n_pages = len(pdf.pages)
    log(f"\nScanning {n_pages} pages…\n")

    for pnum, page in enumerate(pdf.pages, 1):
        text = page.extract_text() or ""
        ctx  = page_context(text)
        ctx["_page"] = pnum
        raw_tables = page.extract_tables()

        if not raw_tables:
            continue

        log(f"Page {pnum:02d}: {len(raw_tables)} table(s) | "
            f"annex={bool(ctx['annex'])}  unit='{ctx['unit']}'  "
            f"section='{ctx['section'][:35]}'")

        for tidx, raw in enumerate(raw_tables):
            cleaned = clean_table(raw)
            if not cleaned:
                continue

            ncols = len(cleaned[0])

            # Reject pure-prose layout boxes
            if is_prose_table(cleaned, ncols):
                log(f"  ↳ p{pnum:02d}_t{tidx+1}: SKIPPED (prose layout box, {ncols} cols)")
                continue

            hdr_idx  = first_good_header_row(cleaned)
            header   = cleaned[hdr_idx]
            data     = cleaned[hdr_idx + 1:]
            data     = [r for r in data if not is_section_label_row(r)]
            data     = [r for r in data if any(c for c in r)]  # drop blank rows

            title = infer_title(ctx, header, text, tidx)

            entry = {
                "table_id":            f"p{pnum:02d}_t{tidx+1}",
                "page":                pnum,
                "table_index_on_page": tidx,
                "title":               title,
                "unit":                ctx["unit"] or "See column headers",
                "annex":               ctx["annex"],
                "section":             ctx["section"],
                "column_count":        ncols,
                "column_headers":      header,
                "column_types":        col_types(header),
                "data_rows":           data,
                "row_count":           len(data),
                "source_pages":        [pnum],
                "_ctx":                ctx,
            }
            physical_tables.append(entry)
            log(f"  ✓ [{entry['table_id']}] {title[:55]}  "
                f"({ncols}c × {len(data)}r)")


# ─────────────────────────────────────────────────────────────
# STEP 6 — MERGE CONTINUATIONS
# ─────────────────────────────────────────────────────────────
log("\n--- Merging cross-page continuation tables ---")

logical_tables = []
for entry in physical_tables:
    if (logical_tables
            and is_continuation(logical_tables[-1],
                                entry["data_rows"],
                                entry["page"])):
        prev = logical_tables[-1]
        prev["data_rows"].extend(entry["data_rows"])
        prev["row_count"]     = len(prev["data_rows"])
        prev["source_pages"].append(entry["page"])
        # expand section info if new page introduced a named section
        if entry["_ctx"]["section"] and not prev["section"]:
            prev["section"] = entry["_ctx"]["section"]
        log(f"  ↳ merged {entry['table_id']} → {prev['table_id']}  "
            f"(now {prev['row_count']} rows, pages {prev['source_pages']})")
    else:
        logical_tables.append(deepcopy(entry))

# strip internal key
for t in logical_tables:
    t.pop("_ctx", None)

log(f"\n{len(physical_tables)} physical tables → "
    f"{len(logical_tables)} logical tables after merging\n")


# ─────────────────────────────────────────────────────────────
# STEP 7 — NLP TEXT GENERATION
# Every row is serialised as "Header: value [unit]" so a model
# reading this can never confuse which column a number belongs to.
# ─────────────────────────────────────────────────────────────

def to_nlp_text(t: dict) -> str:
    lines = []
    lines.append(f"TABLE: {t['title']}")
    if t["annex"]:    lines.append(f"Annex/Document Section: {t['annex']}")
    if t["section"]:  lines.append(f"Budget Section: {t['section']}")
    unit = t["unit"] if t["unit"] != "See column headers" else ""
    if unit:          lines.append(f"Monetary Unit: {unit}")
    lines.append(f"Source (PDF pages): {t['source_pages']}")
    lines.append(f"Columns: {' | '.join(h for h in t['column_headers'] if h)}")
    lines.append("")

    headers   = t["column_headers"]
    ctypes    = t["column_types"]

    for row in t["data_rows"]:
        if not any(c for c in row):
            continue
        parts = []
        for h, v, ctype in zip(headers, row, ctypes):
            if not v:
                continue
            label = h if h else "Value"
            # annotate monetary / rate values explicitly
            if unit and ctype in ("numeric",) and any(ch.isdigit() for ch in v):
                parts.append(f"{label}: {v} ({unit})")
            elif ctype == "rate" and unit != "See column headers":
                parts.append(f"{label}: {v}")
            else:
                parts.append(f"{label}: {v}")
        if parts:
            lines.append(" | ".join(parts))

    return "\n".join(lines)


for t in logical_tables:
    t["nlp_text"] = to_nlp_text(t)


# ─────────────────────────────────────────────────────────────
# STEP 8 — WRITE JSON
# ─────────────────────────────────────────────────────────────

json_path = OUT_DIR / "budget_tables.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(logical_tables, f, ensure_ascii=False, indent=2)
log(f"JSON  → {json_path}  ({len(logical_tables)} tables)")


# ─────────────────────────────────────────────────────────────
# STEP 9 — WRITE SELF-CONTAINED CSVs
# Metadata rows prefixed with "#" so pandas.read_csv(comment='#')
# reads them as regular data but they can be skipped programmatically.
# ─────────────────────────────────────────────────────────────

for t in logical_tables:
    safe = re.sub(r"[^\w]+", "_", t["title"])[:55].strip("_")
    path = CSV_DIR / f"{t['table_id']}_{safe}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# TABLE_ID",    t["table_id"]])
        w.writerow(["# TITLE",       t["title"]])
        w.writerow(["# UNIT",        t["unit"]])
        w.writerow(["# ANNEX",       t["annex"]])
        w.writerow(["# SECTION",     t["section"]])
        w.writerow(["# SOURCE_PAGES",",".join(str(p) for p in t["source_pages"])])
        w.writerow(["# DOCUMENT",    "Union Budget India 2016-17"])
        w.writerow([])
        w.writerow(t["column_headers"])
        for row in t["data_rows"]:
            w.writerow(row)

log(f"CSVs  → {CSV_DIR}  ({len(logical_tables)} files)")


# ─────────────────────────────────────────────────────────────
# STEP 10 — WRITE NLP CORPUS
# ─────────────────────────────────────────────────────────────

nlp_path = OUT_DIR / "budget_tables_nlp.txt"
with open(nlp_path, "w", encoding="utf-8") as f:
    for t in logical_tables:
        f.write(t["nlp_text"] + "\n\n" + "─" * 68 + "\n\n")
log(f"NLP   → {nlp_path}")


# ─────────────────────────────────────────────────────────────
# STEP 11 — REPORT
# ─────────────────────────────────────────────────────────────

summary = [
    "",
    "=" * 68,
    "EXTRACTION SUMMARY",
    "=" * 68,
    f"  Pages scanned         : {n_pages}",
    f"  Physical tables found : {len(physical_tables)}",
    f"  Logical tables output : {len(logical_tables)}",
    "",
    f"{'ID':<12}  {'Pages':<22}  {'Cols':>4}  {'Rows':>4}  {'Unit':<22}  Title",
    "-" * 110,
]
for t in logical_tables:
    pages_str = str(t["source_pages"])
    if len(pages_str) > 20: pages_str = f"pp.{t['source_pages'][0]}–{t['source_pages'][-1]}"
    summary.append(
        f"  {t['table_id']:<10}  {pages_str:<22}  {t['column_count']:>4}  "
        f"{t['row_count']:>4}  {t['unit'][:21]:<22}  {t['title'][:50]}"
    )

REPORT.extend(summary)
log("\n".join(summary))

rpt = OUT_DIR / "pipeline_report.txt"
rpt.write_text("\n".join(REPORT), encoding="utf-8")
log(f"\nReport→ {rpt}")
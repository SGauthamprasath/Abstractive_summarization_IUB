"""
budget_extractor.py
====================
Extracts text from Indian Budget speech PDFs, cleans them, chunks by sentence
boundaries, and outputs a CSV matching the reference dataset schema.

Usage:
    python budget_extractor.py --input_dir ./pdfs --output_csv budget_dataset.csv

Requirements:
    pip install pdfplumber pandas --break-system-packages
"""

import re
import csv
import argparse
from pathlib import Path

import pdfplumber
import pandas as pd


# ---------------------------------------------------------------------------
# 1.  TEXT EXTRACTION
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF. Pages joined with double-newline."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# 2.  ENCODING / MOJIBAKE FIXES
# ---------------------------------------------------------------------------

ENCODING_FIXES = [
    (b"\xe2\x80\x99".decode(), "'"),   # right single quote
    (b"\xe2\x80\x98".decode(), "'"),   # left single quote
    (b"\xe2\x80\x9c".decode(), '"'),   # left double quote
    (b"\xe2\x80\x9d".decode(), '"'),   # right double quote
    (b"\xe2\x80\x94".decode(), "-"),   # em-dash
    (b"\xe2\x80\x93".decode(), "-"),   # en-dash
    (b"\xe2\x80\xa6".decode(), "..."), # ellipsis
    (b"\xc2\xa0".decode(), " "),       # non-breaking space
    # Windows-1252 via latin-1 misread
    ("\x92", "'"),
    ("\x91", "'"),
    ("\x93", '"'),
    ("\x94", '"'),
    ("\x96", "-"),
    ("\x97", "-"),
    # Mojibake sequences
    ("â€™", "'"),
    ("â€˜", "'"),
    ("â€œ", '"'),
    ("â€", '"'),
    ("â€¦", "..."),
    ("\u00e2\u20ac\u2122", "'"),
]

RUPEE_RE = re.compile(r"[\u20b9\u20a8]")          # ₹ or ₨
INR_SPACE_RE = re.compile(r"\bINR\s+(?=\d)")       # INR 500 → Rs. 500
RS_NORM_RE = re.compile(r"\bRs\s*\.\s*(?=\d)")    # normalise Rs . 500 → Rs. 500


def apply_encoding_fixes(text: str) -> str:
    for bad, good in ENCODING_FIXES:
        text = text.replace(bad, good)
    text = RUPEE_RE.sub("Rs.", text)
    text = INR_SPACE_RE.sub("Rs. ", text)
    text = RS_NORM_RE.sub("Rs. ", text)
    return text


# ---------------------------------------------------------------------------
# 3.  STRUCTURAL CLEANING
# ---------------------------------------------------------------------------

# Paragraph-number prefix at start of a line: "12. " or " 2 . "
PARA_NUM_RE = re.compile(r"(?m)^[ \t]*\d{1,3}[ \t]*\.[ \t]+")

# Standalone bare page number: a line that is ONLY 1-3 digits
# Negative lookahead prevents matching "Rs. 500" style lines
PAGE_NUM_RE = re.compile(r"(?m)^[ \t]*\d{1,3}[ \t]*$")

# Hyphenated line-break: "invest-\nment" → "investment"
HYPHEN_BREAK_RE = re.compile(r"-[ \t]*\n[ \t]*([a-z])")

# Multiple blank lines → single blank line  
MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Bullet / list markers at line start
LIST_MARKER_RE = re.compile(r"(?m)^[ \t]*[·•\-]\s+")

BOILERPLATE_LINES = [
    re.compile(r"(?i)^budget\s+\d{4}[-\u2013]\d{2,4}\s+speech"),
    re.compile(r"(?i)^speech\s+of\b"),
    re.compile(r"(?i)^minister\s+of\s+finance\s*$"),
    re.compile(r"(?i)^\d{1,2}(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s*,?\s*\d{4}\s*$"),
    re.compile(r"(?i)^part\s+[ab]\s*$"),
    re.compile(r"(?i)^shri\b.{0,40}$"),   # "Shri P. Chidambaram" standalone line
]


def clean_raw_text(text: str) -> str:
    """Full cleaning pipeline for raw extracted PDF text."""

    # Step 1: encoding fixes
    text = apply_encoding_fixes(text)

    # Step 2: fix hyphenated line-breaks
    text = HYPHEN_BREAK_RE.sub(r"\1", text)

    # Step 3: strip standalone page numbers
    text = PAGE_NUM_RE.sub("", text)

    # Step 4: strip paragraph-number prefixes
    text = PARA_NUM_RE.sub("", text)

    # Step 5: strip bullet markers
    text = LIST_MARKER_RE.sub("", text)

    # Step 6: strip boilerplate lines
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if any(pat.match(stripped) for pat in BOILERPLATE_LINES):
            continue
        clean_lines.append(line)
    text = "\n".join(clean_lines)

    # Step 7: collapse multiple blank lines
    text = MULTI_BLANK_RE.sub("\n\n", text)

    # Step 8: within each line, collapse runs of whitespace
    processed = []
    for line in text.split("\n"):
        processed.append(" ".join(line.split()))
    text = "\n".join(processed)

    return text.strip()


# ---------------------------------------------------------------------------
# 4.  PARAGRAPH RECONSTRUCTION
#     PDF extraction gives us one line per visual line. We need to join
#     continuation lines into proper paragraphs.
# ---------------------------------------------------------------------------

def reconstruct_paragraphs(text: str) -> list[str]:
    """
    Join wrapped lines back into full paragraphs.
    A paragraph ends when:
      - There's a blank line
      - The current line ends with sentence-ending punctuation AND the next line
        starts with a capital letter (new thought)
    Returns a list of paragraph strings.
    """
    raw_lines = text.split("\n")
    paragraphs = []
    current = []

    for i, line in enumerate(raw_lines):
        if not line.strip():
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            # Check if we should start a new paragraph
            if current:
                prev = current[-1]
                # New para if current line starts a section heading (all-caps short)
                if (re.match(r"^[A-Z][A-Z\s&/\-,()]{4,}$", line.strip())
                        and len(line.strip()) < 80):
                    paragraphs.append(" ".join(current))
                    current = [line.strip()]
                    continue
            current.append(line.strip())

    if current:
        paragraphs.append(" ".join(current))

    return [p for p in paragraphs if p.strip()]


# ---------------------------------------------------------------------------
# 5.  SENTENCE SPLITTER
# ---------------------------------------------------------------------------

# Common abbreviations that end with a period but are NOT sentence boundaries
_ABBR = {
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "Hon", "Shri", "Smt",
    "Govt", "No", "viz", "i.e", "e.g", "etc", "vs", "Rs", "approx",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    "Dept", "Distt", "Pvt", "Ltd",
}
_ABBR_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _ABBR) + r")\."
)
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\u2018\u2019])")


def split_sentences(text: str) -> list[str]:
    # Temporarily hide abbreviation dots
    protected = _ABBR_RE.sub(lambda m: m.group(1) + "<<<DOT>>>", text)
    parts = _SENT_BOUNDARY.split(protected)
    sentences = [p.replace("<<<DOT>>>", ".").strip() for p in parts]
    return [s for s in sentences if s]


# ---------------------------------------------------------------------------
# 6.  CHUNKER
# ---------------------------------------------------------------------------

def build_chunks(
    paragraphs: list[str],
    min_words: int = 60,
    max_words: int = 400,
    overlap: int = 1,
) -> list[str]:
    """
    Pack sentences into chunks:
    - Never splits mid-sentence
    - Merges orphan short chunks with preceding chunk
    - Optional sentence overlap for continuity
    """
    all_sents = []
    for para in paragraphs:
        all_sents.extend(split_sentences(para))

    if not all_sents:
        return []

    chunks = []
    current: list[str] = []
    current_wc = 0

    for sent in all_sents:
        wc = len(sent.split())
        if current_wc + wc > max_words and current_wc >= min_words:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            current_wc = sum(len(s.split()) for s in current)
        current.append(sent)
        current_wc += wc

    if current:
        chunks.append(" ".join(current))

    # Merge short orphan chunks with previous
    merged = []
    for chunk in chunks:
        if merged and len(chunk.split()) < min_words:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    # Final: collapse any stray newlines inside a chunk
    merged = [" ".join(c.split()) for c in merged]
    return [c for c in merged if c.strip()]


# ---------------------------------------------------------------------------
# 7.  YEAR PARSING
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(\d{4})[-_](\d{2,4})")


def parse_year(filename: str) -> str:
    m = _YEAR_RE.search(filename)
    if m:
        y1, y2 = m.group(1), m.group(2)
        return f"{y1}_{y2[-2:]}"
    m2 = re.search(r"\d{4}", filename)
    return m2.group(0) if m2 else "unknown"


# ---------------------------------------------------------------------------
# 8.  MAIN
# ---------------------------------------------------------------------------

def process_pdfs(
    input_dir: str,
    output_csv: str,
    min_words: int = 60,
    max_words: int = 400,
    verbose: bool = True,
):
    pdfs = sorted(Path(input_dir).glob("*.pdf"))
    if not pdfs:
        print(f"[WARN] No PDFs found in {input_dir}")
        return

    rows = []
    for pdf_path in pdfs:
        year = parse_year(pdf_path.stem)
        doc_id = f"UB_{year}"
        if verbose:
            print(f"[INFO] {pdf_path.name}  →  {doc_id}", end="  ")

        raw = extract_text_from_pdf(pdf_path)
        cleaned = clean_raw_text(raw)
        paragraphs = reconstruct_paragraphs(cleaned)
        chunks = build_chunks(paragraphs, min_words=min_words, max_words=max_words)

        if verbose:
            print(f"→ {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            rows.append({
                "chunk_id": f"{doc_id}_c{idx}",
                "document_id": doc_id,
                "year": year,
                "chunk_filename": f"chunk_{idx}.txt",
                "chunk_text": chunk,
                "summary_text": "",
            })

    df = pd.DataFrame(rows, columns=[
        "chunk_id", "document_id", "year",
        "chunk_filename", "chunk_text", "summary_text",
    ])
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")

    if verbose:
        print(f"\n[DONE] {len(rows)} total chunks  →  {output_csv}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Extract & chunk Indian Budget PDFs into CSV")
    ap.add_argument("--input_dir",  required=True)
    ap.add_argument("--output_csv", default="budget_dataset.csv")
    ap.add_argument("--min_words",  type=int, default=100)
    ap.add_argument("--max_words",  type=int, default=400)
    ap.add_argument("--quiet",      action="store_true")
    args = ap.parse_args()
    process_pdfs(args.input_dir, args.output_csv, args.min_words, args.max_words, not args.quiet)


if __name__ == "__main__":
    main()
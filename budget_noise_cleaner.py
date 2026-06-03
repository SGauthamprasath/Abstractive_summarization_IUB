"""
budget_noise_cleaner.py
=======================
Second-pass cleaner: removes ceremonial language, filler phrases, and
non-English-script text (while preserving code-switched economic terms
like crore, lakh, Rupees, etc.).

Usage:
    python budget_noise_cleaner.py \
        --input_csv  budget_dataset.csv \
        --output_csv budget_dataset_clean.csv

Requires: pip install pandas langdetect --break-system-packages
"""

import re
import csv
import argparse
import pandas as pd

try:
    from langdetect import detect, LangDetectException
    LANGDETECT = True
except ImportError:
    LANGDETECT = False
    print("[WARN] langdetect not installed — language filter disabled. "
          "Install: pip install langdetect --break-system-packages")


# ---------------------------------------------------------------------------
# 1.  SENTENCE-LEVEL REMOVAL PATTERNS  (whole sentence dropped if matched)
# ---------------------------------------------------------------------------

REMOVE_SENTENCE = [
    # Opening/closing salutations
    re.compile(r"(?i)^(madam|mr\.?|hon['\u2019]?ble|honourable)?\s*(speaker|chair(man|woman|person)?)\s*[,.]?\s*(sir\s*[,.]?)?\s*$"),
    re.compile(r"(?i)^sir\s*[,.]?\s*$"),
    re.compile(r"(?i)^i\s+(rise|stand)\s+to\s+(present|introduce)\s+the\s+budget"),
    re.compile(r"(?i)i\s+commend\s+the\s+budget\s+to\s+(this|the)\s+(august|hon['\u2019]?ble)?\s*house"),
    re.compile(r"(?i)with\s+these\s+words[,.]?\s+i\s+commend"),
    re.compile(r"(?i)^i\s+beg\s+to\s+move\b"),
    re.compile(r"(?i)^thank\s+you\s*[.,!]?\s*$"),
    re.compile(r"(?i)^jai\s+(hind|bharat|india)\s*[.,!]?\s*$"),
    re.compile(r"(?i)^vande\s+mataram\s*[.,!]?\s*$"),
    re.compile(r"(?i)^bharat\s+mata\s+ki\s+jai\s*[.,!]?\s*$"),

    # Ceremonial filler
    re.compile(r"(?i)hon['\u2019]?ble\s+members?\s+will\s+(kindly\s+)?indulge\s+me"),
    re.compile(r"(?i)^permit\s+me\s+to\s+(begin|start)\b"),
    re.compile(r"(?i)^allow\s+me\s+to\s+(begin|start)\b"),

    # Structural headers that carry no content
    re.compile(r"(?i)^budget\s+\d{4}[-\u2013]\d{2,4}\s+speech"),
    re.compile(r"(?i)^speech\s+of\b"),
    re.compile(r"(?i)^minister\s+of\s+finance\s*$"),
    re.compile(r"(?i)^\d{1,2}(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s*,?\s*\d{4}\s*$"),
    re.compile(r"(?i)^part\s+[ab]\s*$"),

    # Sign-off
    re.compile(r"(?i)^\[?\d{1,2}(st|nd|rd|th)?\s+(february|march|july|august)\s*,?\s*\d{4}\]?\s*$"),
]

# ---------------------------------------------------------------------------
# 2.  INLINE PREFIX STRIPS  (remove prefix, keep the rest of the sentence)
# ---------------------------------------------------------------------------

STRIP_PREFIX = [
    # "Mr. Speaker, Sir, ..." or "Madam Speaker, ..."
    re.compile(r"(?i)^(mr\.?\s*)?(madam\s+)?(hon['\u2019]?ble\s+)?(speaker|chair(wo)?man?)\s*,\s*(sir\s*,\s*)?"),
    # "Hon'ble Members," at sentence start
    re.compile(r"(?i)^hon['\u2019]?ble\s+members?\s*,\s*"),
    # "I am glad/pleased/happy to announce that ..."
    re.compile(r"(?i)^(i\s+)?(am\s+)?(glad|pleased|happy|delighted|proud)\s+to\s+(announce|inform|state|say|report)\s+(that|the house that|to the house that)?\s+"),
    # "I now turn to / come to ..."
    re.compile(r"(?i)^i\s+now\s+(turn|come)\s+to\s+"),
    # "Let me now ..."
    re.compile(r"(?i)^let\s+me\s+now\s+"),
    # "I shall now briefly go over ..."
    re.compile(r"(?i)^i\s+shall\s+now\s+(briefly\s+)?(go\s+over|discuss|present)\s+"),
]

# ---------------------------------------------------------------------------
# 3.  NON-LATIN SCRIPT DETECTION
# ---------------------------------------------------------------------------

# Preserve sentences that contain economic content even if mixed-script
_ECONOMIC_KW = re.compile(
    r"\b(crore|lakh|rupee|Rs\.|GDP|fiscal|budget|revenue|tax|deficit|"
    r"growth|inflation|expenditure|allocation|subsidy|duty|tariff|export|import)\b",
    re.IGNORECASE,
)

_NON_ASCII_ALPHA = re.compile(r"[^\x00-\x7F]")


def is_primarily_non_latin(text: str) -> bool:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    non_latin_count = sum(1 for c in alpha if ord(c) > 127)
    return non_latin_count / len(alpha) > 0.45


# ---------------------------------------------------------------------------
# 4.  PER-SENTENCE LOGIC
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\u2018\u2019])")

# Common abbreviations (dots protected during split)
_ABBR_DOTS = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Hon|Shri|Smt|Govt|No|viz|i\.e|e\.g|etc|"
    r"vs|Rs|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Dept|Pvt|Ltd)\."
)


def split_sentences(text: str) -> list[str]:
    protected = _ABBR_DOTS.sub(lambda m: m.group(1) + "<<<DOT>>>", text)
    parts = _SENT_SPLIT.split(protected)
    return [p.replace("<<<DOT>>>", ".").strip() for p in parts if p.strip()]


def clean_sentence(sent: str) -> str | None:
    """
    Returns None if the sentence should be dropped, otherwise the cleaned sentence.
    """
    s = sent.strip()
    if not s:
        return None

    # Check full-removal patterns
    for pat in REMOVE_SENTENCE:
        if pat.search(s):
            return None

    # Drop sentences that are predominantly non-Latin AND lack economic content
    if is_primarily_non_latin(s) and not _ECONOMIC_KW.search(s):
        return None

    # Drop very short sentences without any digits (likely orphan fragments)
    words = s.split()
    if len(words) < 5 and not re.search(r"\d", s):
        return None

    # Strip inline prefixes
    for pat in STRIP_PREFIX:
        new_s = pat.sub("", s).strip()
        if new_s:
            s = new_s
            break  # only strip the first matching prefix

    # Recapitalise if prefix was stripped
    if s and s[0].islower():
        s = s[0].upper() + s[1:]

    return s if s else None


# ---------------------------------------------------------------------------
# 5.  CHUNK CLEANER
# ---------------------------------------------------------------------------

MIN_WORDS = 30   # drop whole chunk if below this after cleaning


def clean_chunk(text: str) -> str:
    """Clean a single chunk. Returns empty string if nothing survives."""
    # Flatten any stray newlines
    text = " ".join(text.split())

    sentences = split_sentences(text)
    cleaned = []
    for sent in sentences:
        result = clean_sentence(sent)
        if result:
            cleaned.append(result)

    out = " ".join(cleaned)
    # Final whitespace normalisation
    return " ".join(out.split())


# ---------------------------------------------------------------------------
# 6.  OPTIONAL: LANGUAGE CHECK ON WHOLE CHUNK
# ---------------------------------------------------------------------------

def is_english(text: str) -> bool:
    if not LANGDETECT:
        return True
    words = text.split()
    # Chunks that are mostly numbers/symbols pass regardless
    numeric = sum(1 for w in words if re.fullmatch(r"[\d.,%()\-+/Rs.]+", w))
    if len(words) and numeric / len(words) > 0.5:
        return True
    try:
        return detect(text) in {"en"}
    except LangDetectException:
        return True  # benefit of the doubt


# ---------------------------------------------------------------------------
# 7.  PIPELINE
# ---------------------------------------------------------------------------

def clean_dataset(input_csv: str, output_csv: str, verbose: bool = True):
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    if "chunk_text" not in df.columns:
        raise ValueError("Input CSV must have 'chunk_text' column")

    total = len(df)
    cleaned_rows = []

    for _, row in df.iterrows():
        original = row["chunk_text"]
        cleaned = clean_chunk(original)

        wc = len(cleaned.split())
        too_short = wc < MIN_WORDS
        non_eng = not is_english(cleaned) if cleaned else True

        if too_short or non_eng:
            reason = "too_short" if too_short else "non_english"
            if verbose:
                preview = cleaned[:80] if cleaned else original[:80]
                print(f"[DROP:{reason}] {row.get('chunk_id','?')} ({wc}w): {preview}...")
            continue

        row = row.copy()
        row["chunk_text"] = cleaned
        cleaned_rows.append(row)

    result_df = pd.DataFrame(cleaned_rows).reset_index(drop=True)

    # Re-index chunk IDs
    counters = {}
    new_ids = []
    for _, row in result_df.iterrows():
        doc = row["document_id"]
        counters[doc] = counters.get(doc, 0)
        new_ids.append(f"{doc}_c{counters[doc]}")
        counters[doc] += 1
    result_df["chunk_id"] = new_ids

    result_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")

    kept = len(result_df)
    dropped = total - kept
    if verbose:
        print(f"\n[DONE]  Input: {total}  |  Dropped: {dropped}  |  Kept: {kept}")
        print(f"        Output: {output_csv}")
    return result_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Remove ceremonial noise / non-English text from budget chunk CSV"
    )
    ap.add_argument("--input_csv",  required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    clean_dataset(args.input_csv, args.output_csv, not args.quiet)


if __name__ == "__main__":
    main()
"""
parsers/pdf_parser.py  –  Robust multi-stage column-aware PDF parser

Pipeline:
    Stage 1 → PyMuPDF  (fastest, handles most PDFs)
    Stage 2 → pdfplumber (handles complex layout / tables)
    Stage 3 → pdfminer.six (handles encrypted / exotic encoding)

Each stage is tried in order; we advance to the next only when the extracted
text is below MIN_CHARS (200).  If all stages fail we return a result with
parse_status = "PARSE_FAILED" so the caller can skip scoring.

Column detection uses bounding-box x-coordinates from PyMuPDF.
"""
import io
from typing import Any, Dict, List, Tuple
from utils.logger import get_logger

log = get_logger("pdf_parser")

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_CHARS: int = 200        # minimum chars to consider extraction a success
COL_THRESHOLD: float = 0.45  # fraction of page width used as column midpoint


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: column splitter
# ═══════════════════════════════════════════════════════════════════════════════

def _sort_blocks_y(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort blocks top→bottom, then left→right (reading order)."""
    return sorted(blocks, key=lambda b: (round(b["y0"] / 20) * 20, b["x0"]))


def _split_columns(
    blocks: List[Dict[str, Any]], page_width: float
) -> Tuple[List[str], List[str]]:
    """
    Classify blocks into LEFT / RIGHT column by x-position vs midpoint.
    Returns (left_texts, right_texts).
    """
    midpoint = page_width * COL_THRESHOLD
    left: List[str] = []
    right: List[str] = []
    for b in _sort_blocks_y(blocks):
        text = b.get("text", "").strip()
        if not text:
            continue
        if b["x0"] < midpoint:
            left.append(text)
        else:
            right.append(text)
    return left, right


def _is_two_column(blocks: List[Dict[str, Any]], page_width: float) -> bool:
    """Heuristic: if horizontal spread > 40 % of page width → two-column."""
    xs = [b["x0"] for b in blocks]
    return bool(xs) and (max(xs) - min(xs)) > page_width * 0.40


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 – PyMuPDF  (primary)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_pymupdf(file_bytes: bytes) -> Tuple[str, str, str, int, bool]:
    """
    Returns (full_text, left_text, right_text, page_count, is_two_column).
    Raises ImportError if fitz not installed.
    Raises RuntimeError if text is empty (caller will fall through).
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    all_left: List[str] = []
    all_right: List[str] = []
    two_col_votes: List[bool] = []

    for page in doc:
        page_width = page.rect.width
        raw_blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)

        blocks: List[Dict[str, Any]] = []
        for b in raw_blocks:
            if b[6] != 0:  # skip image-only blocks
                continue
            text = str(b[4]).strip()
            if text:
                blocks.append({"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3], "text": text})

        two_col_votes.append(_is_two_column(blocks, page_width))
        left, right = _split_columns(blocks, page_width)
        all_left.extend(left)
        all_right.extend(right)

    doc.close()

    page_count = len(two_col_votes)
    two_col    = sum(two_col_votes) > len(two_col_votes) / 2

    left_text  = "\n".join(all_left)
    right_text = "\n".join(all_right)
    full_text  = (left_text + "\n\n" + right_text) if two_col else left_text

    return full_text.strip(), left_text.strip(), right_text.strip(), page_count, two_col


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 – pdfplumber
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_pdfplumber(file_bytes: bytes) -> str:
    """Plain text extraction via pdfplumber (no column split – used as fallback)."""
    import pdfplumber

    parts: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            # Try word-level extraction with bbox for pseudo column detection
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            if words:
                page_width = float(page.width)
                midpoint   = page_width * COL_THRESHOLD
                left_words: List[str]  = []
                right_words: List[str] = []
                for w in sorted(words, key=lambda w: (round(w["top"] / 10) * 10, w["x0"])):
                    if w["x0"] < midpoint:
                        left_words.append(w["text"])
                    else:
                        right_words.append(w["text"])
                parts.append(" ".join(left_words) + "\n" + " ".join(right_words))
            else:
                t = page.extract_text()
                if t:
                    parts.append(t)
    return "\n".join(parts).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 – pdfminer.six  (last resort)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_pdfminer(file_bytes: bytes) -> str:
    """Plain text extraction via pdfminer.six."""
    from pdfminer.high_level import extract_text as pm_extract

    result = pm_extract(io.BytesIO(file_bytes))
    return (result or "").strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf(file_bytes: bytes) -> Dict[str, Any]:
    """
    Multi-stage robust PDF extraction.

    Returns:
        {
          "full_text":      str,
          "left_text":      str,
          "right_text":     str,
          "pages":          int,
          "is_two_column":  bool,
          "parser_used":    str,   # "pymupdf" | "pdfplumber" | "pdfminer"
          "parse_status":   str,   # "OK" | "LOW_CONFIDENCE" | "PARSE_FAILED"
          "char_count":     int,
          "raw_blocks":     list,
        }
    """
    pages = 1
    is_two_col = False

    # ── Stage 1: PyMuPDF ───────────────────────────────────────────────────────
    try:
        full, left, right, pages, is_two_col = _extract_pymupdf(file_bytes)
        if len(full) >= MIN_CHARS:
            log.info("Stage 1 (PyMuPDF) success: chars=%d pages=%d two_col=%s",
                     len(full), pages, is_two_col)
            return _build_result(full, left, right, pages, is_two_col, "pymupdf")
        log.warning("Stage 1 (PyMuPDF) returned only %d chars → trying pdfplumber", len(full))
    except Exception as e:
        log.warning("Stage 1 (PyMuPDF) error: %s", e)

    # ── Stage 2: pdfplumber ────────────────────────────────────────────────────
    try:
        full = _extract_pdfplumber(file_bytes)
        if len(full) >= MIN_CHARS:
            log.info("Stage 2 (pdfplumber) success: chars=%d", len(full))
            return _build_result(full, full, "", pages, False, "pdfplumber")
        log.warning("Stage 2 (pdfplumber) returned only %d chars → trying pdfminer", len(full))
    except Exception as e:
        log.warning("Stage 2 (pdfplumber) error: %s", e)

    # ── Stage 3: pdfminer.six ─────────────────────────────────────────────────
    try:
        full = _extract_pdfminer(file_bytes)
        if len(full) >= MIN_CHARS:
            log.info("Stage 3 (pdfminer) success: chars=%d", len(full))
            return _build_result(full, full, "", pages, False, "pdfminer")
        log.warning("Stage 3 (pdfminer) returned only %d chars", len(full))
    except Exception as e:
        log.warning("Stage 3 (pdfminer) error: %s", e)

    # ── All stages failed ──────────────────────────────────────────────────────
    log.error("⚠ All PDF extraction stages failed for this file")
    return {
        "full_text":     "",
        "left_text":     "",
        "right_text":    "",
        "pages":         pages,
        "is_two_column": False,
        "parser_used":   "none",
        "parse_status":  "PARSE_FAILED",
        "char_count":    0,
        "raw_blocks":    [],
    }


def _build_result(
    full: str, left: str, right: str,
    pages: int, two_col: bool, parser: str,
) -> Dict[str, Any]:
    char_count   = len(full)
    parse_status = "OK" if char_count >= MIN_CHARS else "LOW_CONFIDENCE"
    return {
        "full_text":     full,
        "left_text":     left,
        "right_text":    right,
        "pages":         pages,
        "is_two_column": two_col,
        "parser_used":   parser,
        "parse_status":  parse_status,
        "char_count":    char_count,
        "raw_blocks":    [],
    }


# Keep legacy alias for compatibility
def extract_pdf_pdfplumber_fallback(file_bytes: bytes) -> str:
    return _extract_pdfplumber(file_bytes)

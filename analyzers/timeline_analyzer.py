"""
analyzers/timeline_analyzer.py  –  Candidate Timeline & Growth Analysis

Strategy (cost-optimised):
  - Phase 1 (rule-based): Build chronological event list from education,
    projects, and work_experience using year regex.
  - Phase 2 (LLM, 1 call only): Summarise the ordered events into a human
    narrative and compute a growth_score.

growth_score rubric (0-100):
  - consistency    (no long gaps): up to 30 pts
  - complexity ramp (projects → internships → full-time): up to 40 pts
  - real-world signal (internships / full-time jobs): up to 30 pts
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from utils.logger import get_logger

log = get_logger("timeline_analyzer")

# ── Year extraction helpers ────────────────────────────────────────────────────

_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")


def _extract_years(text: str) -> List[int]:
    """Extract all 4-digit years found in a string."""
    return [int(y) for y in _YEAR_RE.findall(text or "")]


def _earliest_year(text: str) -> Optional[int]:
    years = _extract_years(text)
    return min(years) if years else None


def _latest_year(text: str) -> Optional[int]:
    years = _extract_years(text)
    return max(years) if years else None


# ── Raw event builder (rule-based) ────────────────────────────────────────────

def _build_raw_events(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract raw dated events from education, work_experience, and projects.
    Each event: {"year": int, "type": str, "label": str}
    """
    events: List[Dict[str, Any]] = []

    # --- Education ---
    for edu in candidate.get("education") or []:
        if not isinstance(edu, dict):
            continue
        year_str = str(edu.get("year") or "")
        label    = f"{edu.get('degree','')} at {edu.get('institution','')}".strip(" at")
        # For ongoing degrees use start year; for completed use graduation year
        year = _earliest_year(year_str) if edu.get("is_ongoing") else _latest_year(year_str)
        if year:
            events.append({"year": year, "type": "education", "label": label,
                           "is_ongoing": bool(edu.get("is_ongoing"))})

    # --- Work experience ---
    for exp in candidate.get("work_experience") or []:
        if not isinstance(exp, dict):
            continue
        date_text = f"{exp.get('start','')} {exp.get('end','')}".strip()
        year      = _earliest_year(date_text)
        is_intern = bool(exp.get("is_internship"))
        label     = f"{'Internship' if is_intern else 'Role'}: {exp.get('title','')} at {exp.get('company','')}".strip()
        if year:
            events.append({"year": year, "type": "internship" if is_intern else "fulltime",
                           "label": label, "description": exp.get("description","")})

    # --- Projects ---
    for proj in candidate.get("projects") or []:
        if not isinstance(proj, dict):
            continue
        desc  = proj.get("description","") or ""
        year  = _earliest_year(desc)
        if not year:
            continue
        tech  = ", ".join((proj.get("tech_stack") or [])[:4])
        label = f"Project: {proj.get('name','')} ({tech})"
        events.append({"year": year, "type": "project", "label": label,
                       "description": desc[:120]})

    events.sort(key=lambda e: e["year"])
    return events


# ── Growth score (pure rule-based) ────────────────────────────────────────────

def _compute_growth_score(events: List[Dict[str, Any]]) -> Tuple[float, str]:
    """Return (score 0-100, short_reasoning_str)."""
    if not events:
        return 0.0, "No chronological data found."

    years = [e["year"] for e in events]
    span  = max(years) - min(years) if len(years) > 1 else 0

    # Consistency: penalise gaps > 1 year
    consistency_pts = 30.0
    for i in range(1, len(years)):
        gap = years[i] - years[i - 1]
        if gap > 2:
            consistency_pts = max(consistency_pts - 10, 0)
        elif gap > 1:
            consistency_pts = max(consistency_pts - 5, 0)

    # Complexity ramp: ideal order → education → project → internship → fulltime
    type_order = {"education": 0, "project": 1, "internship": 2, "fulltime": 3}
    types_seq  = [type_order.get(e["type"], 0) for e in events]
    ramp_pts   = 0.0
    for i in range(1, len(types_seq)):
        if types_seq[i] >= types_seq[i - 1]:
            ramp_pts += 40.0 / max(len(types_seq) - 1, 1)
    ramp_pts = min(ramp_pts, 40.0)

    # Real-world signal
    n_fulltime = sum(1 for e in events if e["type"] == "fulltime")
    n_intern   = sum(1 for e in events if e["type"] == "internship")
    rw_pts = min(n_fulltime * 15 + n_intern * 7, 30.0)

    score = round(consistency_pts + ramp_pts + rw_pts, 1)
    reasoning = (
        f"Consistency={consistency_pts:.0f}/30, "
        f"ComplexityRamp={ramp_pts:.0f}/40, "
        f"RealWorldExp={rw_pts:.0f}/30"
    )
    return score, reasoning


# ── LLM narrative (single call, optional) ─────────────────────────────────────

def _llm_growth_summary(events: List[Dict[str, Any]], candidate_name: str) -> str:
    """
    Ask the LLM to narrate the candidate's career arc in 2-3 sentences.
    Returns empty string on failure so callers can gracefully fallback.
    """
    try:
        from extractors.llm_extractor import _chat   # reuse _chat helper
    except ImportError:
        return ""

    event_lines = "\n".join(
        f"- {e['year']}: [{e['type'].upper()}] {e['label']}" for e in events[:12]
    )
    prompt = (
        f"Candidate: {candidate_name}\n\n"
        f"Chronological career events:\n{event_lines}\n\n"
        "Write a 2-3 sentence growth summary for a recruiter. "
        "Describe the candidate's learning progression, key transitions, and career trajectory. "
        "Be factual and concise. Do NOT invent details."
    )
    try:
        result = _chat(prompt, system="You are a concise technical recruiter assistant.")
        return (result.get("content") or "").strip()
    except Exception as e:
        log.warning("LLM growth summary failed: %s", e)
        return ""


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_timeline(
    candidate: Dict[str, Any],
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Entry point.  Given a candidate dict (from build_candidate_dict),
    returns a timeline result dict.

    Output:
    {
        "timeline": [{"year": "2022", "event": "...", "type": "..."}, ...],
        "growth_score": 78.5,
        "growth_summary": "Fast learner...",
        "score_breakdown": "Consistency=28/30 ..."
    }
    """
    name   = candidate.get("name") or "Candidate"
    events = _build_raw_events(candidate)

    timeline = [
        {"year": str(e["year"]), "event": e["label"], "type": e["type"]}
        for e in events
    ]

    growth_score, breakdown = _compute_growth_score(events)

    # 1 LLM call if enabled and we have ≥2 events
    summary = ""
    if use_llm and len(events) >= 2:
        summary = _llm_growth_summary(events, name)

    if not summary:
        # Fallback rule-based summary
        n_proj   = sum(1 for e in events if e["type"] == "project")
        n_intern = sum(1 for e in events if e["type"] == "internship")
        n_ft     = sum(1 for e in events if e["type"] == "fulltime")
        summary  = (
            f"Career spans {len(timeline)} events across education, "
            f"{n_proj} project(s), {n_intern} internship(s), and "
            f"{n_ft} full-time role(s)."
        )

    log.info("Timeline: %d events | growth_score=%.1f", len(events), growth_score)
    return {
        "timeline":        timeline,
        "growth_score":    growth_score,
        "growth_summary":  summary,
        "score_breakdown": breakdown,
    }

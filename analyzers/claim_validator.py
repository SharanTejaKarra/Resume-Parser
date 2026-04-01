"""
analyzers/claim_validator.py  --  Skill Claim Validation & Credibility Scoring

Fully rule-based (zero LLM calls).

Pipeline:
  1. Skill evidence mapping    – per-skill source tracing
  2. Buzzword detection        – buzzword-to-substance ratio
  3. Career switcher detection – domain drift across timeline
  4. Depth mismatch detection  – seniority claims vs evidence
  5. Bonus signal detection    – certs, awards, open source, publications

Credibility score: starts at 50, adjusted by evidence and penalties.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.logger import get_logger
from analyzers.skill_ontology import (
    DOMAIN_ONTOLOGY,
    normalize_skill,
    skill_to_domains,
)

log = get_logger("claim_validator")

# ── Constants ────────────────────────────────────────────────────────────────

BUZZWORDS: List[str] = [
    "synergy", "leverage", "utilize", "paradigm", "ecosystem",
    "holistic", "streamline", "optimize", "empower", "innovative",
    "cutting-edge", "state-of-the-art", "best-in-class", "world-class",
    "dynamic", "proactive",
]

SENIOR_SKILLS: Set[str] = {
    "system design", "architecture", "kubernetes", "team lead",
    "tech lead", "principal", "staff",
}

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Patterns that indicate factual content (numbers, percentages, dates, etc.)
_FACT_PATTERN = re.compile(
    r"\d+%|\d+\.\d+|"           # percentages, decimals
    r"\b\d{4}\b|"               # years
    r"\b\d+[KkMm]?\+?\b|"      # counts like 10K, 5M
    r"\$\d+"                    # dollar amounts
)


# ── 1. Skill evidence mapping ───────────────────────────────────────────────

def _map_skill_evidence(
    candidate: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], float]:
    """
    For each listed skill, trace whether it appears in projects or work.

    Returns:
        (skill_evidence_map, demonstrated_ratio)

    skill_evidence_map example:
        {"Python": {"claimed": True, "demonstrated": True,
                    "sources": ["project:AI Parser", "work:Acme Corp"]}}
    """
    skills: List[str] = list(candidate.get("skills") or [])
    projects: List[Dict[str, Any]] = [
        p for p in (candidate.get("projects") or []) if isinstance(p, dict)
    ]
    work_exp: List[Dict[str, Any]] = [
        e for e in (candidate.get("work_experience") or []) if isinstance(e, dict)
    ]

    evidence: Dict[str, Dict[str, Any]] = {}

    for skill in skills:
        skill_lower = skill.strip().lower()
        sources: List[str] = []

        # Check projects
        for proj in projects:
            tech_stack_lower = [
                t.strip().lower() for t in (proj.get("tech_stack") or [])
            ]
            proj_desc = (proj.get("description") or "").lower()
            proj_name = proj.get("name") or "Unnamed"

            if skill_lower in tech_stack_lower or skill_lower in proj_desc:
                source = f"project:{proj_name}"
                if source not in sources:
                    sources.append(source)

        # Check work experience
        for exp in work_exp:
            text_blob = (
                (exp.get("description") or "") + " " +
                " ".join(exp.get("achievements") or [])
            ).lower()
            company = exp.get("company") or "Unknown"

            if skill_lower in text_blob:
                source = f"work:{company}"
                if source not in sources:
                    sources.append(source)

        evidence[skill] = {
            "claimed": True,
            "demonstrated": len(sources) > 0,
            "sources": sources,
        }

    total = len(evidence)
    demonstrated_count = sum(1 for v in evidence.values() if v["demonstrated"])
    demonstrated_ratio = round(
        demonstrated_count / max(total, 1), 3
    )

    return evidence, demonstrated_ratio


# ── 2. Buzzword detection ───────────────────────────────────────────────────

def _compute_buzzword_ratio(candidate: Dict[str, Any]) -> float:
    """
    Compute ratio of buzzwords to total description words.

    Scans work_experience descriptions/achievements and project descriptions.
    """
    parts: List[str] = []

    for exp in (candidate.get("work_experience") or []):
        if not isinstance(exp, dict):
            continue
        parts.append(exp.get("description") or "")
        parts.extend(exp.get("achievements") or [])

    for proj in (candidate.get("projects") or []):
        if not isinstance(proj, dict):
            continue
        parts.append(proj.get("description") or "")

    combined = " ".join(parts).lower()
    words = combined.split()
    total_words = len(words)

    if total_words == 0:
        return 0.0

    buzzword_count = 0
    for bw in BUZZWORDS:
        buzzword_count += combined.count(bw.lower())

    return round(buzzword_count / total_words, 4)


# ── 3. Career switcher detection ────────────────────────────────────────────

def _detect_career_switch(
    candidate: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Detect if the candidate has shifted domains across their career.

    Looks at work_experience + education + projects chronologically,
    maps skills to domains, and checks if earliest vs latest domains diverge.

    Returns (is_career_switcher, ordered_career_domains).
    """
    # Build chronological entries with associated skills/domains
    entries: List[Tuple[int, Set[str]]] = []

    # Education
    for edu in (candidate.get("education") or []):
        if not isinstance(edu, dict):
            continue
        year_str = str(edu.get("year") or "")
        years = _YEAR_RE.findall(year_str)
        year = int(min(years)) if years else None
        if year is None:
            continue
        field = (edu.get("field") or "").lower()
        domains: Set[str] = set()
        # Map education field to domains heuristically
        for domain, skills_list in DOMAIN_ONTOLOGY.items():
            for s in skills_list:
                if s.lower() in field:
                    domains.add(domain)
                    break
        if domains:
            entries.append((year, domains))

    # Work experience
    for exp in (candidate.get("work_experience") or []):
        if not isinstance(exp, dict):
            continue
        start_str = str(exp.get("start") or "")
        years = _YEAR_RE.findall(start_str)
        year = int(min(years)) if years else None
        if year is None:
            continue
        # Extract domains from description and title
        text_blob = (
            (exp.get("title") or "") + " " +
            (exp.get("description") or "") + " " +
            " ".join(exp.get("achievements") or [])
        ).lower()
        domains = set()
        for domain, skills_list in DOMAIN_ONTOLOGY.items():
            for s in skills_list:
                if s.lower() in text_blob:
                    domains.add(domain)
                    break
        if domains:
            entries.append((year, domains))

    # Projects
    for proj in (candidate.get("projects") or []):
        if not isinstance(proj, dict):
            continue
        desc = (proj.get("description") or "").lower()
        years = _YEAR_RE.findall(desc)
        year = int(min(years)) if years else None
        # If no year in description, put at end
        if year is None:
            year = 9999
        tech_stack = [t.lower() for t in (proj.get("tech_stack") or [])]
        domains = set()
        for tech in tech_stack:
            for d in skill_to_domains(tech):
                domains.add(d)
        if domains:
            entries.append((year, domains))

    if not entries:
        return False, []

    # Sort chronologically
    entries.sort(key=lambda e: e[0])

    # Build ordered domain list (preserving first-seen order)
    ordered_domains: List[str] = []
    for _, doms in entries:
        for d in sorted(doms):
            if d not in ordered_domains:
                ordered_domains.append(d)

    # Compare earliest and latest domain sets
    if len(entries) < 2:
        return False, ordered_domains

    earliest_domains = entries[0][1]
    latest_domains = entries[-1][1]

    # Compute overlap ratio
    all_domains = earliest_domains | latest_domains
    overlap = earliest_domains & latest_domains

    if len(all_domains) == 0:
        return False, ordered_domains

    overlap_ratio = len(overlap) / len(all_domains)
    is_switcher = overlap_ratio < 0.5

    return is_switcher, ordered_domains


# ── 4. Depth mismatch detection ─────────────────────────────────────────────

def _check_depth_mismatches(
    candidate: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Detect mismatches between claimed skill depth and actual evidence.

    Flags:
      - Student/fresher claiming senior-level skills
      - Skills claimed with experience duration but thin project evidence
    """
    mismatches: List[Dict[str, str]] = []
    candidate_type = str(candidate.get("candidate_type") or "").lower()
    skills: List[str] = list(candidate.get("skills") or [])
    projects: List[Dict[str, Any]] = [
        p for p in (candidate.get("projects") or []) if isinstance(p, dict)
    ]

    # Student/fresher with senior skills
    if candidate_type in ("student", "fresher"):
        for skill in skills:
            if skill.lower() in SENIOR_SKILLS:
                mismatches.append({
                    "skill": skill,
                    "claimed_level": "senior",
                    "evidence_level": candidate_type,
                })

    # Check for skills with thin project evidence
    # Build a map: skill -> number of projects that use it
    skill_project_count: Dict[str, int] = {}
    for skill in skills:
        count = 0
        for proj in projects:
            tech_lower = [t.lower() for t in (proj.get("tech_stack") or [])]
            proj_desc = (proj.get("description") or "").lower()
            if skill.lower() in tech_lower or skill.lower() in proj_desc:
                count += 1
        skill_project_count[skill] = count

    # If candidate claims significant experience but has minimal project evidence
    ft_years = float(candidate.get("full_time_experience_years") or 0)
    if ft_years >= 5:
        for skill in skills:
            if skill_project_count.get(skill, 0) <= 1:
                # Only flag for "core" skills that would normally appear in projects
                domains = skill_to_domains(skill)
                if domains:  # only flag mapped skills, not soft skills
                    mismatches.append({
                        "skill": skill,
                        "claimed_level": f"{ft_years:.0f}yr experience",
                        "evidence_level": (
                            f"{skill_project_count[skill]} project(s)"
                        ),
                    })

    return mismatches


# ── 5. Bonus signal detection ───────────────────────────────────────────────

def _detect_bonus_signals(
    candidate: Dict[str, Any],
    resume_text: str,
) -> Dict[str, Any]:
    """
    Detect positive credibility signals: certifications, achievements,
    open source contributions, publications.
    """
    certs: List[str] = list(candidate.get("certifications") or [])
    achievements: List[str] = list(candidate.get("achievements") or [])

    # Open source detection
    text_lower = resume_text.lower() if resume_text else ""
    # Also scan project descriptions for open-source signals
    for proj in (candidate.get("projects") or []):
        if isinstance(proj, dict):
            text_lower += " " + (proj.get("description") or "").lower()
            url = (proj.get("url") or "").lower()
            text_lower += " " + url

    open_source_signals = [
        "open source", "open-source", "contributor", "contributed to",
        "github.com", "pull request", "merged pr", "maintainer",
    ]
    has_open_source = any(sig in text_lower for sig in open_source_signals)

    # Publication detection
    pub_signals = [
        "published", "publication", "paper", "conference",
        "journal", "arxiv", "ieee", "acm", "springer",
        "speaking", "speaker", "talk at", "presented at",
        "keynote", "workshop",
    ]
    pub_count = sum(1 for sig in pub_signals if sig in text_lower)
    # Rough heuristic: each unique signal found counts as ~1 publication
    # but cap at a reasonable number
    publications = min(pub_count, 5)

    return {
        "certifications": len(certs),
        "achievements": len(achievements),
        "open_source": has_open_source,
        "publications": publications,
    }


# ── 6. Metric flag extraction ───────────────────────────────────────────────

def _extract_metric_flags(
    candidate: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Identify specific metric claims that seem questionable.
    Returns list of {"claim": str, "issue": str, "severity": str}.
    """
    flags: List[Dict[str, str]] = []
    pct_re = re.compile(r"(\d+(?:\.\d+)?)\s*%")

    all_text_chunks: List[str] = []
    for exp in (candidate.get("work_experience") or []):
        if isinstance(exp, dict):
            all_text_chunks.append(exp.get("description") or "")
            all_text_chunks.extend(exp.get("achievements") or [])
    for proj in (candidate.get("projects") or []):
        if isinstance(proj, dict):
            all_text_chunks.append(proj.get("description") or "")

    for chunk in all_text_chunks:
        for match in pct_re.finditer(chunk):
            pct_val = float(match.group(1))
            if pct_val > 95 and pct_val != 100:
                flags.append({
                    "claim": chunk[:100].strip(),
                    "issue": f"Improvement claim of {match.group(0)} seems inflated",
                    "severity": "medium",
                })
            elif pct_val > 80:
                flags.append({
                    "claim": chunk[:100].strip(),
                    "issue": f"Large improvement claim ({match.group(0)}) -- verify",
                    "severity": "low",
                })

    return flags


# ── Credibility score ────────────────────────────────────────────────────────

def _compute_credibility_score(
    demonstrated_ratio: float,
    buzzword_ratio: float,
    bonus_signals: Dict[str, Any],
    depth_mismatches: List[Dict[str, str]],
    metric_flags: List[Dict[str, str]],
) -> float:
    """
    Compute credibility score.

    Base: 50
    + up to 30 for demonstrated_ratio
    + up to 10 for low buzzword ratio
    + up to 10 for bonus signals
    - 5 per depth mismatch
    - 3 per metric flag
    Clamped to 0-100.
    """
    score = 50.0

    # Demonstrated skills: up to +30
    score += demonstrated_ratio * 30.0

    # Buzzword ratio: up to +10 (lower is better)
    # 0 buzzwords = +10, ratio >= 0.05 = +0
    buzzword_penalty = min(buzzword_ratio / 0.05, 1.0)
    score += (1.0 - buzzword_penalty) * 10.0

    # Bonus signals: up to +10
    bonus_pts = 0.0
    bonus_pts += min(bonus_signals.get("certifications", 0) * 2.0, 4.0)
    bonus_pts += min(bonus_signals.get("achievements", 0) * 1.5, 3.0)
    if bonus_signals.get("open_source"):
        bonus_pts += 2.0
    bonus_pts += min(bonus_signals.get("publications", 0) * 1.0, 2.0)
    score += min(bonus_pts, 10.0)

    # Penalties
    score -= len(depth_mismatches) * 5.0
    score -= len(metric_flags) * 3.0

    return round(max(min(score, 100.0), 0.0), 1)


# ── Public API ───────────────────────────────────────────────────────────────

def validate_claims(
    candidate: Dict[str, Any],
    resume_text: str = "",
) -> Dict[str, Any]:
    """
    Main entry point for claim validation and credibility assessment.

    Runs skill evidence mapping, buzzword analysis, career switch detection,
    depth mismatch checks, and bonus signal detection.

    Args:
        candidate: Parsed candidate dict from LLM extraction.
        resume_text: Optional raw resume text for bonus signal scanning.

    Returns:
        {
            "credibility_score": float,
            "skill_evidence": {...},
            "demonstrated_ratio": float,
            "metric_flags": [...],
            "buzzword_ratio": float,
            "is_career_switcher": bool,
            "career_domains": [...],
            "depth_mismatches": [...],
            "bonus_signals": {...},
        }
    """
    # 1. Skill evidence
    skill_evidence, demonstrated_ratio = _map_skill_evidence(candidate)

    # 2. Buzzwords
    buzzword_ratio = _compute_buzzword_ratio(candidate)

    # 3. Career switch
    is_career_switcher, career_domains = _detect_career_switch(candidate)

    # 4. Depth mismatches
    depth_mismatches = _check_depth_mismatches(candidate)

    # 5. Bonus signals
    bonus_signals = _detect_bonus_signals(candidate, resume_text)

    # 6. Metric flags
    metric_flags = _extract_metric_flags(candidate)

    # Credibility score
    credibility_score = _compute_credibility_score(
        demonstrated_ratio=demonstrated_ratio,
        buzzword_ratio=buzzword_ratio,
        bonus_signals=bonus_signals,
        depth_mismatches=depth_mismatches,
        metric_flags=metric_flags,
    )

    log.info(
        "Claims: credibility=%.1f demo_ratio=%.2f buzzwords=%.4f "
        "switcher=%s domains=%d mismatches=%d",
        credibility_score, demonstrated_ratio, buzzword_ratio,
        is_career_switcher, len(career_domains), len(depth_mismatches),
    )

    return {
        "credibility_score": credibility_score,
        "skill_evidence": skill_evidence,
        "demonstrated_ratio": demonstrated_ratio,
        "metric_flags": metric_flags,
        "buzzword_ratio": buzzword_ratio,
        "is_career_switcher": is_career_switcher,
        "career_domains": career_domains,
        "depth_mismatches": depth_mismatches,
        "bonus_signals": bonus_signals,
    }

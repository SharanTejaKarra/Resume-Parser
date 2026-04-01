"""
scoring/level_router.py  –  Experience-level classifier & weight router

Classifies candidates into experience levels and provides
level-appropriate evaluation weights + scoring paths.

Levels:
  junior   (0-2 yrs effective)  -> proof_of_work path
  mid      (2-5 yrs)            -> hybrid path
  senior   (5-10 yrs)           -> business_impact path
  lead     (10+ yrs or titles)  -> business_impact path
"""
from typing import Any, Dict, List

from utils.logger import get_logger

log = get_logger("level_router")

# ── Weight tables per level ──────────────────────────────────────────────────

_LEVEL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "junior": {
        "jd_similarity": 0.35,
        "skill_match":   0.25,
        "projects":      0.25,
        "github":        0.08,
        "leetcode":      0.07,
        "experience":    0.00,
    },
    "mid": {
        "jd_similarity": 0.35,
        "skill_match":   0.25,
        "projects":      0.15,
        "github":        0.05,
        "leetcode":      0.05,
        "experience":    0.15,
    },
    "senior": {
        "jd_similarity": 0.40,
        "skill_match":   0.20,
        "projects":      0.10,
        "github":        0.05,
        "leetcode":      0.00,
        "experience":    0.25,
    },
    "lead": {
        "jd_similarity": 0.40,
        "skill_match":   0.15,
        "projects":      0.05,
        "github":        0.00,
        "leetcode":      0.00,
        "experience":    0.40,
    },
}

_EVAL_PATHS: Dict[str, str] = {
    "junior": "proof_of_work",
    "mid":    "hybrid",
    "senior": "business_impact",
    "lead":   "business_impact",
}

# ── Title keyword sets ───────────────────────────────────────────────────────

_LEAD_KEYWORDS   = {"lead", "principal", "staff", "architect", "director", "head"}
_SENIOR_KEYWORDS = {"senior", "manager"}
_JUNIOR_KEYWORDS = {"intern", "trainee", "junior", "associate"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _effective_years(candidate: Dict[str, Any]) -> float:
    """Compute effective years of experience from full-time + internships."""
    ft_years       = float(candidate.get("full_time_exp_years", 0) or 0)
    intern_months  = float(candidate.get("internship_months", 0) or 0)
    intern_years   = intern_months / 12.0
    return ft_years + intern_years


def _extract_titles(candidate: Dict[str, Any]) -> List[str]:
    """Collect all job titles from work_experience entries."""
    titles: List[str] = []
    for exp in candidate.get("work_experience") or []:
        if isinstance(exp, dict):
            title = (exp.get("title") or "").strip().lower()
            if title:
                titles.append(title)
    return titles


def _title_signal(titles: List[str]) -> str:
    """
    Return "lead_boost", "senior_boost", "junior_anchor", or "neutral"
    based on title keywords.

    Lead keywords  (lead/principal/staff/architect/director/head) → "lead_boost"
    Senior keywords (senior/manager) → "senior_boost"
    Junior keywords (intern/trainee/junior/associate) → "junior_anchor"
    """
    for title in titles:
        tokens = set(title.split())
        if tokens & _LEAD_KEYWORDS:
            return "lead_boost"
    for title in titles:
        tokens = set(title.split())
        if tokens & _SENIOR_KEYWORDS:
            return "senior_boost"
    for title in titles:
        tokens = set(title.split())
        if tokens & _JUNIOR_KEYWORDS:
            return "junior_anchor"
    return "neutral"


def _is_student_or_fresher(candidate: Dict[str, Any]) -> bool:
    """Check if candidate is flagged as student or fresher."""
    ctype = (candidate.get("candidate_type") or "").lower()
    return ctype in ("student", "fresher") or candidate.get("is_student", False)


def _has_exceptional_portfolio(candidate: Dict[str, Any]) -> bool:
    """
    Fresher with strong GitHub score or 3+ substantial projects
    may qualify for mid-level evaluation.
    """
    gh_score  = float(candidate.get("github_score", 0) or 0)
    projects  = candidate.get("projects") or []
    rich_proj = sum(
        1 for p in projects
        if isinstance(p, dict) and len(p.get("tech_stack") or []) >= 2
    )
    return gh_score >= 60 or rich_proj >= 3


# ── Public API ───────────────────────────────────────────────────────────────

def classify_candidate_level(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify a candidate into an experience level and return
    evaluation path, recommended weights, and reasoning.

    Returns:
    {
        "level": str,
        "confidence": float,
        "evaluation_path": str,
        "level_weights": Dict[str, float],
        "reasoning": str,
    }
    """
    eff_years   = _effective_years(candidate)
    titles      = _extract_titles(candidate)
    title_sig   = _title_signal(titles)
    is_fresher  = _is_student_or_fresher(candidate)

    # ── Step 1: base level from years ────────────────────────────────────
    if eff_years >= 10:
        level = "lead"
    elif eff_years >= 5:
        level = "senior"
    elif eff_years >= 2:
        level = "mid"
    else:
        level = "junior"

    # ── Step 2: title-based adjustments ──────────────────────────────────
    if title_sig == "lead_boost":
        # Explicit lead/principal/staff/architect/director/head → push up one tier
        if level == "junior":
            level = "mid"
        elif level == "mid":
            level = "senior"
        elif level == "senior":
            level = "lead"
        # lead stays lead

    elif title_sig == "senior_boost":
        # "senior" / "manager" confirms senior but doesn't push to lead
        if level == "junior":
            level = "mid"
        elif level == "mid":
            level = "senior"
        # senior stays senior, lead stays lead

    if title_sig == "junior_anchor":
        if level == "mid":
            level = "junior"
        # don't demote senior/lead on anchor keywords alone

    # ── Step 3: fresher with exceptional portfolio → mid ─────────────────
    if is_fresher and level == "junior" and _has_exceptional_portfolio(candidate):
        level = "mid"

    # ── Step 4: confidence ───────────────────────────────────────────────
    years_fits_bracket = (
        (level == "junior" and eff_years < 2)
        or (level == "mid"    and 2 <= eff_years < 5)
        or (level == "senior" and 5 <= eff_years < 10)
        or (level == "lead"   and eff_years >= 10)
    )
    title_aligns = (
        (title_sig == "lead_boost"   and level == "lead")
        or (title_sig == "senior_boost" and level in ("senior", "lead"))
        or (title_sig == "junior_anchor" and level == "junior")
        or (title_sig == "neutral")
    )

    if years_fits_bracket and title_aligns:
        confidence = 0.9
    elif years_fits_bracket or title_aligns:
        confidence = 0.7
    else:
        confidence = 0.5

    # ── Step 5: build result ─────────────────────────────────────────────
    weights = _LEVEL_WEIGHTS[level]
    path    = _EVAL_PATHS[level]

    reasoning = (
        f"Effective experience: {eff_years:.1f} yrs | "
        f"Title signal: {title_sig} | "
        f"Fresher: {is_fresher} | "
        f"Classified as '{level}' (confidence {confidence:.1f})"
    )

    log.info(
        "Level classification: %s (conf=%.1f, path=%s, eff_yrs=%.1f)",
        level, confidence, path, eff_years,
    )
    return {
        "level":           level,
        "confidence":      confidence,
        "evaluation_path": path,
        "level_weights":   weights,
        "reasoning":       reasoning,
    }


def compute_level_adjusted_score(
    candidate:  Dict[str, Any],
    base_ats:   Dict[str, Any],
    level_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Re-compute the ATS score using level-appropriate weights.

    Parameters
    ----------
    candidate  : full candidate dict
    base_ats   : output of compute_ats_score()
    level_info : output of classify_candidate_level()

    Returns
    -------
    {
        "level_adjusted_score": float,
        "base_ats_score": float,
        "level": str,
        "evaluation_path": str,
        "weight_adjustments": Dict[str, float],
    }
    """
    base_score   = float(base_ats.get("ats_score", 0))
    base_weights = base_ats.get("weights") or {}
    lw           = level_info["level_weights"]

    # Normalised component scores (0-1) from the ATS breakdown
    jd_sim   = float(base_ats.get("jd_similarity", 0))
    skill    = float(base_ats.get("skill_match_pct", 0)) / 100.0
    proj     = float(base_ats.get("project_score", 0))   / 100.0
    gh       = float(base_ats.get("github_score", 0))    / 100.0
    lc       = float(base_ats.get("leetcode_score", 0))   / 100.0
    exp      = float(base_ats.get("experience_score", 0)) / 100.0

    adjusted = (
        lw["jd_similarity"] * jd_sim
        + lw["skill_match"] * skill
        + lw["projects"]    * proj
        + lw["github"]      * gh
        + lw["leetcode"]    * lc
        + lw["experience"]  * exp
    )
    adjusted_score = round(float(adjusted * 100), 1)

    # Weight deltas
    weight_adjustments: Dict[str, float] = {}
    for key in lw:
        old = float(base_weights.get(key, 0))
        diff = round(lw[key] - old, 4)
        if diff != 0.0:
            weight_adjustments[key] = diff

    log.info(
        "Level-adjusted score: %.1f (base=%.1f, level=%s, path=%s)",
        adjusted_score, base_score, level_info["level"], level_info["evaluation_path"],
    )
    return {
        "level_adjusted_score": adjusted_score,
        "base_ats_score":       base_score,
        "level":                level_info["level"],
        "evaluation_path":      level_info["evaluation_path"],
        "weight_adjustments":   weight_adjustments,
    }

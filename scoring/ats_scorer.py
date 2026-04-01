"""
scoring/ats_scorer.py  –  ATS composite score calculator

Weights (student/fresher-aware):
─────────────────────────────────────────────────
Standard formula:
  0.40  JD similarity      (embedding cosine)
  0.25  skill match        (fuzzy embed)
  0.15  projects           (count + richness)
  0.10  github             (activity score /100)
  0.05  leetcode           (weighted solve /100)
  0.05  experience         (full-time + 1.0×intern)
─────────────────────────────────────────────────
Fresher / student override (full_time_exp < 1yr):
  0.40  JD similarity
  0.30  skill match
  0.20  projects
  0.05  github
  0.05  leetcode
  0.00  experience         (not penalized for 0 years)
─────────────────────────────────────────────────
Score normalised to 0–100.
"""
from typing import Any, Dict, List
from utils.logger import get_logger

log = get_logger("ats_scorer")


def _project_score(projects: List[Any]) -> float:
    """Score 0-1 based on project count and tech-stack richness."""
    if not projects:
        return 0.0
    score = 0.0
    for p in projects[:6]:
        if isinstance(p, dict):
            tech_count = len(p.get("tech_stack") or [])
            desc_len   = len(p.get("description") or "")
            score = score + min(0.2 + tech_count * 0.05 + (desc_len / 500.0) * 0.1, 0.3)
    return min(score, 1.0)


def _experience_score(
    full_time_years: float,
    internship_months: float,
    min_required: float,
    is_student: bool,
) -> float:
    """
    Compute experience score 0-1 using weighted model.
    Internships count at 1.0× weight (same as full-time).
    Students who only have internships are not penalized.
    """
    if is_student and full_time_years < 1:
        # For students: experience score = 1.0 (not penalized)
        return 1.0

    intern_weighted = (internship_months / 12.0) * 1.0
    effective       = full_time_years + intern_weighted
    if min_required <= 0:
        min_required = 2.0
    return min(effective / min_required, 1.0)


def compute_ats_score(
    jd_similarity:      float,   # 0–1
    skill_match_pct:    float,   # 0–100
    full_time_exp_years: float,  # only full-time jobs
    internship_months:  float,   # internship duration in months
    is_student:         bool,
    candidate_type:     str,     # "student" | "fresher" | "experienced"
    min_exp_required:   float,   # from JD
    github_score:       float,   # 0–100
    leetcode_score:     float,   # 0–100
    projects:           List[Any],
) -> Dict[str, Any]:
    """Compute weighted ATS score and return full breakdown."""

    jd_sim_score  = float(jd_similarity)
    skill_score   = float(skill_match_pct) / 100.0
    gh_norm       = float(github_score)  / 100.0
    lc_norm       = float(leetcode_score) / 100.0
    proj_score    = _project_score(projects)

    is_fresher    = (candidate_type in ("student", "fresher")) or (float(full_time_exp_years) < 1.0)
    exp_score_raw = _experience_score(
        float(full_time_exp_years), float(internship_months),
        float(min_exp_required), is_student,
    )

    if is_fresher:
        # Fresher weights: experience removed, more weight on skills + projects
        weighted = (
            0.40 * jd_sim_score +
            0.30 * skill_score  +
            0.20 * proj_score   +
            0.05 * gh_norm      +
            0.05 * lc_norm
        )
        weights_used = {
            "jd_similarity": 0.40,
            "skill_match":   0.30,
            "projects":      0.20,
            "github":        0.05,
            "leetcode":      0.05,
            "experience":    0.00,
        }
    else:
        # Standard weights
        weighted = (
            0.40 * jd_sim_score +
            0.25 * skill_score  +
            0.15 * proj_score   +
            0.10 * gh_norm      +
            0.05 * lc_norm      +
            0.05 * exp_score_raw
        )
        weights_used = {
            "jd_similarity": 0.40,
            "skill_match":   0.25,
            "projects":      0.15,
            "github":        0.10,
            "leetcode":      0.05,
            "experience":    0.05,
        }

    intern_weighted_yrs = round(float(internship_months) / 12.0 * 1.0, 2)
    effective_exp       = round(float(full_time_exp_years) + intern_weighted_yrs, 2)
    ats                 = round(float(weighted * 100), 1)

    breakdown: Dict[str, Any] = {
        "ats_score":              ats,
        "scoring_mode":           "fresher" if is_fresher else "standard",
        "jd_similarity":          round(float(jd_similarity), 4),
        "skill_match_pct":        float(skill_match_pct),
        "project_score":          round(float(proj_score * 100), 1),
        "experience_score":       round(float(exp_score_raw * 100), 1),
        "github_score":           float(github_score),
        "leetcode_score":         float(leetcode_score),
        # experience detail
        "full_time_years":        float(full_time_exp_years),
        "internship_months":      float(internship_months),
        "intern_weighted_years":  intern_weighted_yrs,
        "effective_exp_years":    effective_exp,
        "is_student":             is_student,
        "candidate_type":         candidate_type,
        "weights":                weights_used,
    }

    log.info(
        "ATS %s-mode: %.1f | sim=%.2f skill=%.1f%% proj=%.1f gh=%.1f "
        "ft=%.1fy intern=%gmo eff=%.2fy",
        "fresher" if is_fresher else "standard",
        ats, jd_similarity, skill_match_pct, proj_score * 100,
        github_score, full_time_exp_years, internship_months, effective_exp,
    )
    return breakdown


# ── Confidence scoring ───────────────────────────────────────────────────────

def compute_confidence_score(
    ats_breakdown:      Dict[str, Any],
    consistency_result: Dict[str, Any],
    claim_result:       Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute overall confidence in the ATS score by combining
    consistency analysis, claim validation, and data completeness.

    Parameters
    ----------
    ats_breakdown      : output of compute_ats_score()
    consistency_result : output of consistency checker (expects "score" key, 0-100)
    claim_result       : output of claim validator   (expects "credibility_score" key, 0-100)

    Returns
    -------
    {
        "confidence_level": str,   # "high" | "medium" | "low"
        "confidence_score": float, # 0-100
        "factors": {
            "consistency": float,
            "credibility": float,
            "data_completeness": float,
        },
        "recommendation": str,
    }
    """
    # ── Factor 1: consistency (weight 0.4) ───────────────────────────────
    consistency = float(consistency_result.get("score", 0) or 0)

    # ── Factor 2: credibility (weight 0.4) ───────────────────────────────
    credibility = float(claim_result.get("credibility_score", 0) or 0)

    # ── Factor 3: data completeness (weight 0.2) ────────────────────────
    completeness_fields = [
        float(ats_breakdown.get("github_score", 0) or 0) > 0,
        float(ats_breakdown.get("leetcode_score", 0) or 0) > 0,
        float(ats_breakdown.get("skill_match_pct", 0) or 0) > 0,
        float(ats_breakdown.get("project_score", 0) or 0) > 0,
        float(ats_breakdown.get("jd_similarity", 0) or 0) > 0,
        float(ats_breakdown.get("experience_score", 0) or 0) > 0,
    ]
    data_completeness = (sum(completeness_fields) / len(completeness_fields)) * 100.0

    # ── Weighted confidence score ────────────────────────────────────────
    confidence_score = round(
        0.4 * consistency + 0.4 * credibility + 0.2 * data_completeness,
        1,
    )

    # ── Confidence level ─────────────────────────────────────────────────
    if confidence_score >= 75:
        confidence_level = "high"
    elif confidence_score >= 50:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    # ── Recommendation ───────────────────────────────────────────────────
    ats_score = float(ats_breakdown.get("ats_score", 0) or 0)

    if ats_score >= 70 and confidence_score >= 70:
        recommendation = "strong_candidate"
    elif ats_score >= 50 and confidence_score >= 50:
        recommendation = "review_recommended"
    elif ats_score >= 50 and confidence_score < 50:
        recommendation = "proceed_with_caution"
    else:
        recommendation = "flag_for_review"

    log.info(
        "Confidence: %.1f (%s) | ATS=%.1f | rec=%s | "
        "consistency=%.1f credibility=%.1f completeness=%.1f",
        confidence_score, confidence_level, ats_score, recommendation,
        consistency, credibility, data_completeness,
    )
    return {
        "confidence_level": confidence_level,
        "confidence_score": confidence_score,
        "factors": {
            "consistency":      round(consistency, 1),
            "credibility":      round(credibility, 1),
            "data_completeness": round(data_completeness, 1),
        },
        "recommendation": recommendation,
    }

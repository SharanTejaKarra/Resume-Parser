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
    Internships count at 0.3× weight.
    Students who only have internships are not penalized.
    """
    if is_student and full_time_years < 1:
        # For students: experience score = 1.0 (not penalized)
        return 1.0

    intern_weighted = (internship_months / 12.0) * 0.3
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

    intern_weighted_yrs = round(float(internship_months) / 12.0 * 0.3, 2)
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

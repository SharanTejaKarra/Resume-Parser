"""
analyzers/consistency_checker.py  --  Resume Consistency & Red-Flag Detection

Fully rule-based (zero LLM calls).

Pipeline:
  1. Timeline consistency    – overlapping dates, implausible experience claims
  2. Skill-experience align  – undemonstrated skills, skill padding
  3. Metric realism          – inflated percentages, missing quantification
  4. AI-generated content    – overused AI-tell phrases, uniformity signals
  5. Employment gap analysis – gaps between roles, after education, career breaks

Consistency score: starts at 100, deducted per red flag severity.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

log = get_logger("consistency_checker")

# ── Constants ────────────────────────────────────────────────────────────────

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

_AI_TELL_PHRASES: List[str] = [
    "leveraged", "utilized", "spearheaded", "orchestrated",
    "cutting-edge", "state-of-the-art", "seamlessly",
    "robust and scalable", "innovative solutions",
]

_SENIOR_SKILLS: set = {
    "kubernetes", "system design", "architecture", "team lead",
    "tech lead", "principal", "staff", "distributed systems",
    "microservices architecture", "infrastructure",
}

_SEVERITY_DEDUCTIONS: Dict[str, int] = {
    "high":   15,
    "medium": 8,
    "low":    3,
}

# Common action verbs that AI tools use to start every bullet
_ACTION_VERBS: set = {
    "developed", "implemented", "designed", "built", "created",
    "managed", "led", "spearheaded", "orchestrated", "engineered",
    "architected", "optimized", "enhanced", "improved", "delivered",
    "established", "executed", "launched", "streamlined", "integrated",
    "deployed", "configured", "automated", "maintained", "collaborated",
}

# Percentage pattern: captures things like "500%", "99.5%", "by 200%"
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


# ── Year extraction helpers ──────────────────────────────────────────────────

def _extract_years(text: str) -> List[int]:
    """Extract all 4-digit years from a string."""
    return [int(y) for y in _YEAR_RE.findall(text or "")]


def _earliest_year(text: str) -> Optional[int]:
    years = _extract_years(text)
    return min(years) if years else None


def _latest_year(text: str) -> Optional[int]:
    years = _extract_years(text)
    return max(years) if years else None


# ── 1. Timeline consistency ──────────────────────────────────────────────────

def _check_timeline(candidate: Dict[str, Any]) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Validate timeline coherence across education, work, and projects.

    Returns (red_flags, warnings).
    """
    red_flags: List[Dict[str, str]] = []
    warnings: List[str] = []

    # Collect education years
    edu_end_years: List[int] = []
    edu_ongoing = False
    for edu in (candidate.get("education") or []):
        if not isinstance(edu, dict):
            continue
        year_str = str(edu.get("year") or "")
        if edu.get("is_ongoing"):
            edu_ongoing = True
        end_y = _latest_year(year_str)
        if end_y:
            edu_end_years.append(end_y)

    latest_edu_end = max(edu_end_years) if edu_end_years else None

    # Collect work experience date ranges
    work_ranges: List[Dict[str, Any]] = []
    for exp in (candidate.get("work_experience") or []):
        if not isinstance(exp, dict):
            continue
        start_str = str(exp.get("start") or "")
        end_str = str(exp.get("end") or "")
        start_y = _earliest_year(start_str)
        end_y = _latest_year(end_str) or _latest_year(start_str)
        is_intern = bool(exp.get("is_internship"))
        work_ranges.append({
            "start": start_y,
            "end": end_y,
            "is_internship": is_intern,
            "title": exp.get("title", ""),
            "company": exp.get("company", ""),
        })

    # Check: claimed experience vs graduation year
    ft_exp_years = float(candidate.get("full_time_experience_years") or 0)
    if latest_edu_end and ft_exp_years > 0:
        years_since_grad = 2026 - latest_edu_end  # current year reference
        if ft_exp_years > years_since_grad + 1:
            red_flags.append({
                "type": "implausible_experience",
                "severity": "high",
                "detail": (
                    f"Claims {ft_exp_years:.0f}yr full-time experience but "
                    f"graduated ~{latest_edu_end}, only {years_since_grad}yr ago"
                ),
            })

    # Check: education end overlapping full-time work start (internships OK)
    if latest_edu_end:
        for wr in work_ranges:
            if wr["is_internship"] or wr["start"] is None:
                continue
            if wr["start"] < latest_edu_end:
                red_flags.append({
                    "type": "education_work_overlap",
                    "severity": "medium",
                    "detail": (
                        f"Full-time role '{wr['title']}' at {wr['company']} "
                        f"starts {wr['start']} but education ends {latest_edu_end}"
                    ),
                })

    # Check: employment gaps > 12 months between consecutive roles
    sorted_ranges = sorted(
        [r for r in work_ranges if r["start"] is not None],
        key=lambda r: r["start"],
    )
    for i in range(1, len(sorted_ranges)):
        prev_end = sorted_ranges[i - 1].get("end") or sorted_ranges[i - 1]["start"]
        curr_start = sorted_ranges[i]["start"]
        if prev_end is not None and curr_start is not None:
            gap_years = curr_start - prev_end
            if gap_years > 1:
                red_flags.append({
                    "type": "large_employment_gap",
                    "severity": "medium",
                    "detail": (
                        f"Gap of ~{gap_years} year(s) between "
                        f"'{sorted_ranges[i-1]['title']}' (ended ~{prev_end}) and "
                        f"'{sorted_ranges[i]['title']}' (started ~{curr_start})"
                    ),
                })

    return red_flags, warnings


# ── 2. Skill-experience alignment ───────────────────────────────────────────

def _check_skill_alignment(
    candidate: Dict[str, Any],
    resume_text: str,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Check whether listed skills are backed by projects or experience.

    Returns (red_flags, warnings).
    """
    red_flags: List[Dict[str, str]] = []
    warnings: List[str] = []

    skills: List[str] = list(candidate.get("skills") or [])
    projects: List[Dict[str, Any]] = list(candidate.get("projects") or [])
    work_exp: List[Dict[str, Any]] = list(candidate.get("work_experience") or [])
    candidate_type = str(candidate.get("candidate_type") or "").lower()

    # Skill padding: many skills, zero evidence
    if len(skills) >= 15 and len(projects) == 0 and len(work_exp) == 0:
        red_flags.append({
            "type": "skill_padding",
            "severity": "high",
            "detail": (
                f"Lists {len(skills)} skills but has 0 projects and "
                f"0 work experience entries"
            ),
        })

    # Build set of skills demonstrated in projects and work
    demonstrated: set = set()

    for proj in projects:
        if not isinstance(proj, dict):
            continue
        for tech in (proj.get("tech_stack") or []):
            demonstrated.add(tech.strip().lower())
        # Also scan project descriptions
        desc = (proj.get("description") or "").lower()
        for skill in skills:
            if skill.lower() in desc:
                demonstrated.add(skill.lower())

    for exp in work_exp:
        if not isinstance(exp, dict):
            continue
        text_blob = (
            (exp.get("description") or "") + " " +
            " ".join(exp.get("achievements") or [])
        ).lower()
        for skill in skills:
            if skill.lower() in text_blob:
                demonstrated.add(skill.lower())

    # Undemonstrated skills
    undemonstrated = [s for s in skills if s.lower() not in demonstrated]
    if undemonstrated and len(undemonstrated) > len(skills) * 0.5:
        warnings.append(
            f"{len(undemonstrated)}/{len(skills)} skills are not demonstrated "
            f"in any project or work description: "
            f"{', '.join(undemonstrated[:8])}"
        )

    # Senior skills claimed by students
    if candidate_type in ("student", "fresher"):
        senior_claimed = [
            s for s in skills
            if s.lower() in _SENIOR_SKILLS
        ]
        if senior_claimed:
            warnings.append(
                f"Student/fresher claims advanced skills typically "
                f"requiring experience: {', '.join(senior_claimed)}"
            )

    return red_flags, warnings


# ── 3. Metric realism ───────────────────────────────────────────────────────

def _check_metric_realism(
    candidate: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Scan descriptions and achievements for inflated or missing metrics.

    Returns (red_flags, warnings).
    """
    red_flags: List[Dict[str, str]] = []
    warnings: List[str] = []

    all_descriptions: List[str] = []
    all_achievements: List[str] = []

    for exp in (candidate.get("work_experience") or []):
        if not isinstance(exp, dict):
            continue
        desc = exp.get("description") or ""
        if desc:
            all_descriptions.append(desc)
        for ach in (exp.get("achievements") or []):
            all_achievements.append(ach)

    for proj in (candidate.get("projects") or []):
        if not isinstance(proj, dict):
            continue
        desc = proj.get("description") or ""
        if desc:
            all_descriptions.append(desc)

    combined_text = " ".join(all_descriptions + all_achievements)

    # Check for inflated percentage claims
    for text_chunk in all_achievements + all_descriptions:
        for match in _PERCENT_RE.finditer(text_chunk):
            pct_value = float(match.group(1))
            if pct_value > 95 and pct_value != 100:
                red_flags.append({
                    "type": "inflated_metric",
                    "severity": "low",
                    "detail": (
                        f"Possibly inflated claim: '{match.group(0)}' "
                        f"in '{text_chunk[:80]}...'"
                    ),
                })

    # Check for no quantifiable metrics at all
    has_any_number = bool(re.search(r"\d+", combined_text))
    if (all_descriptions or all_achievements) and not has_any_number:
        warnings.append(
            "No quantifiable metrics found in any work description or "
            "achievement. Adding specific numbers strengthens credibility."
        )

    return red_flags, warnings


# ── 4. AI-generated content detection ────────────────────────────────────────

def _detect_ai_content(
    candidate: Dict[str, Any],
    resume_text: str,
) -> Tuple[float, List[Dict[str, str]], List[str]]:
    """
    Estimate probability that resume content is AI-generated.

    Returns (ai_risk_score 0.0-1.0, red_flags, warnings).
    """
    red_flags: List[Dict[str, str]] = []
    warnings: List[str] = []

    # Gather all text: resume_text or reconstruct from candidate fields
    if resume_text.strip():
        full_text = resume_text.lower()
    else:
        parts: List[str] = []
        for exp in (candidate.get("work_experience") or []):
            if isinstance(exp, dict):
                parts.append(exp.get("description") or "")
                parts.extend(exp.get("achievements") or [])
        for proj in (candidate.get("projects") or []):
            if isinstance(proj, dict):
                parts.append(proj.get("description") or "")
        full_text = " ".join(parts).lower()

    words = full_text.split()
    total_words = len(words)
    if total_words == 0:
        return 0.0, red_flags, warnings

    # Count AI-tell phrases
    phrase_count = 0
    for phrase in _AI_TELL_PHRASES:
        phrase_count += full_text.count(phrase.lower())

    ratio = phrase_count / total_words

    if ratio > 0.04:
        ai_risk = min(0.9, 0.6 + (ratio - 0.04) * 10)
        red_flags.append({
            "type": "ai_generated_content",
            "severity": "medium",
            "detail": (
                f"High density of AI-typical phrases "
                f"({phrase_count} occurrences in {total_words} words, "
                f"ratio={ratio:.4f})"
            ),
        })
    elif ratio > 0.02:
        ai_risk = 0.3 + (ratio - 0.02) * 15
        warnings.append(
            f"Moderate AI-content signal: {phrase_count} AI-typical phrases "
            f"detected (ratio={ratio:.4f})"
        )
    else:
        ai_risk = ratio * 15  # low, scales from 0 to ~0.3

    # Check bullet point uniformity: do all bullets start with action verbs?
    bullets: List[str] = []
    for exp in (candidate.get("work_experience") or []):
        if isinstance(exp, dict):
            bullets.extend(exp.get("achievements") or [])
    for proj in (candidate.get("projects") or []):
        if isinstance(proj, dict):
            desc = proj.get("description") or ""
            # Split on bullet-like patterns
            for line in desc.split("\n"):
                line = line.strip().lstrip("-*").strip()
                if line:
                    bullets.append(line)

    if len(bullets) >= 4:
        action_start_count = sum(
            1 for b in bullets
            if b.split()[0].lower().rstrip("ed").rstrip("s") in _ACTION_VERBS
            or b.split()[0].lower() in _ACTION_VERBS
        )
        uniformity = action_start_count / len(bullets)
        if uniformity >= 1.0:
            ai_risk = min(ai_risk + 0.15, 1.0)
            warnings.append(
                f"All {len(bullets)} bullet points start with an action verb "
                f"-- perfect uniformity suggests AI-generated content"
            )

    ai_risk = round(min(max(ai_risk, 0.0), 1.0), 3)
    return ai_risk, red_flags, warnings


# ── 5. Employment gap analysis ───────────────────────────────────────────────

def _analyze_gaps(
    candidate: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build year-by-year timeline from work_experience and identify gaps > 6 months.

    Returns list of gap dicts:
      {"from_year": int, "to_year": int, "gap_months": int, "type": str}
    """
    gaps: List[Dict[str, Any]] = []

    # Collect education year ranges (to exclude gaps during education)
    edu_years: set = set()
    for edu in (candidate.get("education") or []):
        if not isinstance(edu, dict):
            continue
        year_str = str(edu.get("year") or "")
        years = _extract_years(year_str)
        if len(years) >= 2:
            for y in range(min(years), max(years) + 1):
                edu_years.add(y)
        elif len(years) == 1:
            # Single year -- assume education covers at least that year
            edu_years.add(years[0])

    # Build sorted work timeline entries
    work_entries: List[Dict[str, Optional[int]]] = []
    for exp in (candidate.get("work_experience") or []):
        if not isinstance(exp, dict):
            continue
        start_str = str(exp.get("start") or "")
        end_str = str(exp.get("end") or "")
        start_y = _earliest_year(start_str)
        end_y = _latest_year(end_str)
        if start_y is None:
            continue
        if end_y is None:
            end_y = start_y
        work_entries.append({"start": start_y, "end": end_y})

    if len(work_entries) < 2:
        # Check for gap after education if there is one work entry
        if len(work_entries) == 1 and edu_years:
            latest_edu = max(edu_years)
            first_work_start = work_entries[0]["start"]
            if first_work_start is not None and first_work_start - latest_edu > 0:
                gap_months = (first_work_start - latest_edu) * 12
                if gap_months > 6:
                    gaps.append({
                        "from_year": latest_edu,
                        "to_year": first_work_start,
                        "gap_months": gap_months,
                        "type": "after_education",
                    })
        return gaps

    work_entries.sort(key=lambda e: e["start"] or 0)

    for i in range(1, len(work_entries)):
        prev_end = work_entries[i - 1]["end"] or work_entries[i - 1]["start"]
        curr_start = work_entries[i]["start"]
        if prev_end is None or curr_start is None:
            continue

        gap_months = (curr_start - prev_end) * 12
        if gap_months <= 6:
            continue

        # Skip if gap years fall entirely within education
        gap_during_education = all(
            y in edu_years
            for y in range(prev_end, curr_start + 1)
        )
        if gap_during_education:
            continue

        gap_type = "between_roles"
        # If the gap starts right after education ends
        if prev_end in edu_years or (prev_end + 1) in edu_years:
            gap_type = "after_education"

        gaps.append({
            "from_year": prev_end,
            "to_year": curr_start,
            "gap_months": gap_months,
            "type": gap_type,
        })

    return gaps


# ── Score computation ────────────────────────────────────────────────────────

def _compute_consistency_score(red_flags: List[Dict[str, str]]) -> float:
    """Start at 100, deduct per red flag severity. Minimum 0."""
    score = 100.0
    for flag in red_flags:
        severity = flag.get("severity", "low")
        score -= _SEVERITY_DEDUCTIONS.get(severity, 3)
    return max(round(score, 1), 0.0)


# ── Public API ───────────────────────────────────────────────────────────────

def check_resume_consistency(
    candidate: Dict[str, Any],
    resume_text: str = "",
) -> Dict[str, Any]:
    """
    Main entry point for resume consistency analysis.

    Runs all checks (timeline, skill alignment, metric realism,
    AI content detection, employment gaps) and returns a consolidated report.

    Args:
        candidate: Parsed candidate dict from LLM extraction.
        resume_text: Optional raw resume text for AI-content analysis.

    Returns:
        {
            "consistency_score": float,   # 0-100
            "red_flags": [...],
            "warnings": [...],
            "timeline_gaps": [...],
            "ai_content_risk": float,     # 0.0-1.0
            "summary": str,
        }
    """
    all_red_flags: List[Dict[str, str]] = []
    all_warnings: List[str] = []

    # 1. Timeline
    timeline_flags, timeline_warns = _check_timeline(candidate)
    all_red_flags.extend(timeline_flags)
    all_warnings.extend(timeline_warns)

    # 2. Skill alignment
    skill_flags, skill_warns = _check_skill_alignment(candidate, resume_text)
    all_red_flags.extend(skill_flags)
    all_warnings.extend(skill_warns)

    # 3. Metric realism
    metric_flags, metric_warns = _check_metric_realism(candidate)
    all_red_flags.extend(metric_flags)
    all_warnings.extend(metric_warns)

    # 4. AI content
    ai_risk, ai_flags, ai_warns = _detect_ai_content(candidate, resume_text)
    all_red_flags.extend(ai_flags)
    all_warnings.extend(ai_warns)

    # 5. Employment gaps
    timeline_gaps = _analyze_gaps(candidate)

    # Compute score
    consistency_score = _compute_consistency_score(all_red_flags)

    # Build summary
    high_count = sum(1 for f in all_red_flags if f["severity"] == "high")
    med_count = sum(1 for f in all_red_flags if f["severity"] == "medium")
    low_count = sum(1 for f in all_red_flags if f["severity"] == "low")

    if high_count > 0:
        summary = (
            f"Resume has {high_count} high-severity issue(s) requiring attention; "
            f"consistency score {consistency_score}/100."
        )
    elif med_count > 0:
        summary = (
            f"Resume has {med_count} medium-severity flag(s); "
            f"consistency score {consistency_score}/100."
        )
    elif all_warnings:
        summary = (
            f"No red flags but {len(all_warnings)} warning(s) noted; "
            f"consistency score {consistency_score}/100."
        )
    else:
        summary = f"Resume appears consistent; score {consistency_score}/100."

    log.info(
        "Consistency: score=%.1f flags=%d warnings=%d gaps=%d ai_risk=%.2f",
        consistency_score, len(all_red_flags), len(all_warnings),
        len(timeline_gaps), ai_risk,
    )

    return {
        "consistency_score": consistency_score,
        "red_flags": all_red_flags,
        "warnings": all_warnings,
        "timeline_gaps": timeline_gaps,
        "ai_content_risk": ai_risk,
        "summary": summary,
    }

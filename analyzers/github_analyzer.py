"""
analyzers/github_analyzer.py  –  GitHub profile enrichment
Fetches repos, languages, stars — no scraping.
Uses GitHub REST API v3 (unauthenticated = 60 req/hr, authenticated = 5000).
"""
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import requests
from utils.logger import get_logger

log = get_logger("github_analyzer")

BASE_URL = "https://api.github.com"


def _headers() -> Dict[str, str]:
    token = os.getenv("GITHUB_TOKEN", "")
    if token and not token.startswith("your_"):
        return {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
    return {"Accept": "application/vnd.github.v3+json"}


def analyze_github(username: str) -> Dict[str, Any]:
    """
    Fetch public GitHub data for a username.
    Returns enriched profile metrics and a computed activity score.
    """
    if not username:
        return _empty_result("no username provided")

    log.info("Fetching GitHub profile for: %s", username)

    try:
        user_resp = requests.get(
            f"{BASE_URL}/users/{username}", headers=_headers(), timeout=10
        )
        if user_resp.status_code == 404:
            return _empty_result(f"user {username} not found")
        if user_resp.status_code != 200:
            return _empty_result(f"API error {user_resp.status_code}")

        user: Dict[str, Any] = user_resp.json()

        # ── repos ──────────────────────────────────────────────────────────────
        repos_resp = requests.get(
            f"{BASE_URL}/users/{username}/repos",
            params={"per_page": 100, "sort": "updated"},
            headers=_headers(),
            timeout=10,
        )
        repos: List[Dict[str, Any]] = (
            repos_resp.json() if repos_resp.status_code == 200 else []
        )

        repo_count  = len(repos)
        total_stars = sum(int(r.get("stargazers_count") or 0) for r in repos)
        total_forks = sum(int(r.get("forks_count") or 0) for r in repos)

        # ── language frequency ─────────────────────────────────────────────────
        lang_freq: Dict[str, int] = {}
        for r in repos:
            lang = r.get("language")
            if lang and isinstance(lang, str):
                lang_freq[lang] = lang_freq.get(lang, 0) + 1

        top_langs: List[str] = sorted(
            lang_freq.keys(), key=lambda k: lang_freq[k], reverse=True
        )[:5]

        # ── featured repos (most stars) ────────────────────────────────────────
        featured_sorted = sorted(
            repos,
            key=lambda r: int(r.get("stargazers_count") or 0),
            reverse=True,
        )[:5]
        featured_list: List[Dict[str, Any]] = [
            {
                "name":        r.get("name", ""),
                "description": r.get("description", "") or "",
                "stars":       int(r.get("stargazers_count") or 0),
                "language":    r.get("language") or "",
                "url":         r.get("html_url", ""),
            }
            for r in featured_sorted
        ]

        # ── score ──────────────────────────────────────────────────────────────
        followers = int(user.get("followers") or 0)
        score = _compute_score(repo_count, total_stars, total_forks, followers)

        result: Dict[str, Any] = {
            "username":       username,
            "name":           user.get("name", "") or "",
            "bio":            user.get("bio", "") or "",
            "followers":      followers,
            "following":      int(user.get("following") or 0),
            "public_repos":   repo_count,
            "total_stars":    total_stars,
            "total_forks":    total_forks,
            "top_languages":  top_langs,
            "featured_repos": featured_list,
            "github_score":   score,
            "profile_url":    f"https://github.com/{username}",
            "error":          None,
        }
        log.info(
            "GitHub done: repos=%d stars=%d score=%.1f",
            repo_count, total_stars, score,
        )
        return result

    except requests.exceptions.RequestException as e:
        log.warning("GitHub fetch error: %s", e)
        return _empty_result(str(e))


def _compute_score(repos: int, stars: int, forks: int, followers: int) -> float:
    """
    Normalised GitHub activity score 0–100.
    Formula: repos*2 + stars*1.5 + forks + followers*0.5, capped at 100.
    """
    raw = float(repos) * 2.0 + float(stars) * 1.5 + float(forks) + float(followers) * 0.5
    return min(round(raw, 1), 100.0)


def _empty_result(reason: str) -> Dict[str, Any]:
    return {
        "username":      None,
        "name":          None,
        "bio":           None,
        "followers":     0,
        "following":     0,
        "public_repos":  0,
        "total_stars":   0,
        "total_forks":   0,
        "top_languages": [],
        "featured_repos": [],
        "github_score":  0.0,
        "profile_url":   None,
        "error":         reason,
    }


# ── v2: Deep GitHub analysis ──────────────────────────────────────────────────


def _compute_deep_score(
    base_score: float,
    commit_patterns: Dict[str, Any],
    code_quality: Dict[str, Any],
    tech_complexity: str,
) -> float:
    """
    Weighted combination of base score and v2 signals.
    deep_score = base*0.5 + commit_pattern*0.2 + code_quality*0.15 + complexity*0.15
    Each component is normalised to 0–100 before weighting.
    """
    # Commit pattern score (0-100)
    cp_score = 0.0
    if commit_patterns.get("has_consistent_activity"):
        cp_score += 50.0
    if commit_patterns.get("recent_activity"):
        cp_score += 30.0
    avg_commits = commit_patterns.get("avg_commits_per_repo", 0.0)
    cp_score += min(avg_commits * 2.0, 20.0)  # up to 20 pts for avg commits

    # Code quality score (0-100)
    cq_score = 0.0
    readme_repos = code_quality.get("has_readme_repos", 0)
    ci_repos = code_quality.get("has_ci_repos", 0)
    lang_diversity = code_quality.get("language_diversity", 0)
    cq_score += min(readme_repos * 10.0, 40.0)
    cq_score += min(ci_repos * 15.0, 30.0)
    if code_quality.get("uses_multiple_languages"):
        cq_score += 15.0
    cq_score += min(lang_diversity * 3.0, 15.0)

    # Complexity score (0-100)
    complexity_map = {"basic": 25.0, "intermediate": 55.0, "advanced": 90.0}
    cx_score = complexity_map.get(tech_complexity, 25.0)

    deep = (
        base_score * 0.5
        + cp_score * 0.2
        + cq_score * 0.15
        + cx_score * 0.15
    )
    return min(round(deep, 1), 100.0)


def analyze_github_deep(username: str) -> Dict[str, Any]:
    """
    Enhanced GitHub analysis: calls analyze_github() then layers on
    commit patterns, code quality signals, repo quality, and tech complexity.
    """
    base_result = analyze_github(username)

    # Short-circuit if the base call failed
    if base_result.get("error"):
        base_result.update({
            "commit_patterns": {
                "has_consistent_activity": False,
                "recent_activity": False,
                "avg_commits_per_repo": 0.0,
            },
            "code_quality_signals": {
                "has_readme_repos": 0,
                "has_ci_repos": 0,
                "uses_multiple_languages": False,
                "language_diversity": 0,
            },
            "repo_quality": [],
            "tech_complexity": "basic",
            "deep_score": 0.0,
        })
        return base_result

    try:
        log.info("Running deep GitHub analysis for: %s", username)
        now = datetime.now(timezone.utc)

        # ── fetch repos (full list) for deeper inspection ─────────────────
        repos_resp = requests.get(
            f"{BASE_URL}/users/{username}/repos",
            params={"per_page": 100, "sort": "updated"},
            headers=_headers(),
            timeout=10,
        )
        repos: List[Dict[str, Any]] = (
            repos_resp.json() if repos_resp.status_code == 200 else []
        )

        repo_count = len(repos)

        # ── commit patterns ───────────────────────────────────────────────
        pushed_dates: List[datetime] = []
        for r in repos:
            pushed_at = r.get("pushed_at")
            if pushed_at:
                try:
                    pushed_dates.append(
                        datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
                    )
                except (ValueError, TypeError):
                    pass

        created_dates: List[datetime] = []
        for r in repos:
            created_at = r.get("created_at")
            if created_at:
                try:
                    created_dates.append(
                        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    )
                except (ValueError, TypeError):
                    pass

        # Consistent activity: repos created across multiple weeks (not all same week)
        has_consistent = False
        if created_dates:
            weeks = {d.isocalendar()[:2] for d in created_dates}  # (year, week) tuples
            has_consistent = len(weeks) > max(1, repo_count // 3)

        # Recent activity: any repo pushed in last 6 months
        six_months_ago = now.timestamp() - (180 * 86400)
        recent_activity = any(
            d.timestamp() > six_months_ago for d in pushed_dates
        )

        # Average commits per repo — approximate via push count heuristic
        # (no extra API call; use pushed_at spread as proxy)
        avg_commits_per_repo = 0.0
        if repo_count > 0:
            repos_with_pushes = sum(1 for r in repos if r.get("pushed_at"))
            avg_commits_per_repo = round(
                (repos_with_pushes / repo_count) * 10.0, 1
            )  # rough heuristic: pushed repos * 10 / total

        commit_patterns = {
            "has_consistent_activity": has_consistent,
            "recent_activity": recent_activity,
            "avg_commits_per_repo": avg_commits_per_repo,
        }

        # ── code quality signals ──────────────────────────────────────────
        has_readme_repos = sum(
            1 for r in repos if r.get("description") and r["description"].strip()
        )
        # CI detection: repos whose name or description hint at CI config
        ci_patterns = {"ci", "cd", "actions", "pipeline", "workflow", "deploy"}
        has_ci_repos = 0
        for r in repos:
            name_lower = (r.get("name") or "").lower()
            desc_lower = (r.get("description") or "").lower()
            topics = [t.lower() for t in (r.get("topics") or [])]
            combined = name_lower + " " + desc_lower + " " + " ".join(topics)
            if any(pat in combined for pat in ci_patterns):
                has_ci_repos += 1

        lang_freq: Dict[str, int] = {}
        for r in repos:
            lang = r.get("language")
            if lang and isinstance(lang, str):
                lang_freq[lang] = lang_freq.get(lang, 0) + 1

        language_diversity = len(lang_freq)
        uses_multiple = language_diversity >= 2

        code_quality = {
            "has_readme_repos": has_readme_repos,
            "has_ci_repos": has_ci_repos,
            "uses_multiple_languages": uses_multiple,
            "language_diversity": language_diversity,
        }

        # ── repo quality (top 5) ─────────────────────────────────────────
        one_year_ago = now.timestamp() - (365 * 86400)
        repo_quality_list: List[Dict[str, Any]] = []

        for r in repos:
            signals: List[str] = []
            score = 0.0

            if r.get("description") and r["description"].strip():
                signals.append("has_description")
                score += 20.0

            if not r.get("fork"):
                signals.append("not_fork")
                score += 20.0

            if int(r.get("stargazers_count") or 0) > 0:
                signals.append("has_stars")
                score += 20.0

            # Recent update
            pushed_at = r.get("pushed_at")
            if pushed_at:
                try:
                    pt = datetime.fromisoformat(
                        pushed_at.replace("Z", "+00:00")
                    ).timestamp()
                    if pt > one_year_ago:
                        signals.append("recent_update")
                        score += 20.0
                except (ValueError, TypeError):
                    pass

            if r.get("homepage") or r.get("topics"):
                signals.append("has_homepage_or_topics")
                score += 20.0

            repo_quality_list.append({
                "name": r.get("name", ""),
                "quality_score": score,
                "signals": signals,
            })

        # Sort by quality score descending, take top 5
        repo_quality_list.sort(key=lambda x: x["quality_score"], reverse=True)
        repo_quality = repo_quality_list[:5]

        # ── tech complexity ───────────────────────────────────────────────
        max_stars = max(
            (int(r.get("stargazers_count") or 0) for r in repos), default=0
        )
        if (language_diversity >= 5 and repo_count >= 15) or max_stars >= 10:
            tech_complexity = "advanced"
        elif language_diversity >= 3 or 5 <= repo_count <= 15:
            tech_complexity = "intermediate"
        else:
            tech_complexity = "basic"

        # ── deep score ────────────────────────────────────────────────────
        deep_score = _compute_deep_score(
            base_result["github_score"],
            commit_patterns,
            code_quality,
            tech_complexity,
        )

        base_result.update({
            "commit_patterns": commit_patterns,
            "code_quality_signals": code_quality,
            "repo_quality": repo_quality,
            "tech_complexity": tech_complexity,
            "deep_score": deep_score,
        })

        log.info(
            "Deep analysis done: complexity=%s deep_score=%.1f",
            tech_complexity, deep_score,
        )
        return base_result

    except Exception as exc:
        log.warning("Deep analysis error, falling back to base result: %s", exc)
        base_result.update({
            "commit_patterns": {
                "has_consistent_activity": False,
                "recent_activity": False,
                "avg_commits_per_repo": 0.0,
            },
            "code_quality_signals": {
                "has_readme_repos": 0,
                "has_ci_repos": 0,
                "uses_multiple_languages": False,
                "language_diversity": 0,
            },
            "repo_quality": [],
            "tech_complexity": "basic",
            "deep_score": base_result.get("github_score", 0.0),
        })
        return base_result

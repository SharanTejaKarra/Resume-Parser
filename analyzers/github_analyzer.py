"""
analyzers/github_analyzer.py  –  GitHub profile enrichment
Fetches repos, languages, stars — no scraping.
Uses GitHub REST API v3 (unauthenticated = 60 req/hr, authenticated = 5000).
"""
import os
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

"""
analyzers/leetcode_analyzer.py  –  LeetCode public stats via GraphQL API
No auth required for public profiles.
"""
from typing import Any, Dict, List
import requests
from utils.logger import get_logger

log = get_logger("leetcode_analyzer")

GRAPHQL_URL = "https://leetcode.com/graphql"

QUERY = """
query getUserProfile($username: String!) {
  matchedUser(username: $username) {
    username
    profile {
      realName
      ranking
    }
    submitStats {
      acSubmissionNum {
        difficulty
        count
      }
    }
    badges {
      name
    }
  }
}
"""


def analyze_leetcode(username: str) -> Dict[str, Any]:
    """Fetch LeetCode public stats for a username."""
    if not username:
        return _empty_result("no username provided")

    log.info("Fetching LeetCode stats for: %s", username)

    try:
        resp = requests.post(
            GRAPHQL_URL,
            json={"query": QUERY, "variables": {"username": username}},
            headers={
                "Content-Type": "application/json",
                "Referer":      "https://leetcode.com",
            },
            timeout=15,
        )

        if resp.status_code != 200:
            return _empty_result(f"HTTP {resp.status_code}")

        payload: Dict[str, Any] = resp.json()
        data = (payload.get("data") or {}).get("matchedUser")
        if not data:
            return _empty_result(f"user '{username}' not found")

        submit_stats = data.get("submitStats") or {}
        ac_list: List[Dict[str, Any]] = submit_stats.get("acSubmissionNum") or []

        ac: Dict[str, int] = {
            str(s.get("difficulty", "")): int(s.get("count") or 0)
            for s in ac_list
        }
        easy   = ac.get("Easy",   0)
        medium = ac.get("Medium", 0)
        hard   = ac.get("Hard",   0)
        total  = easy + medium + hard

        score = _compute_score(easy, medium, hard)
        badges_raw: List[Dict[str, Any]] = data.get("badges") or []
        badges: List[str] = [str(b.get("name", "")) for b in badges_raw]

        profile = data.get("profile") or {}
        result: Dict[str, Any] = {
            "username":       username,
            "real_name":      str(profile.get("realName") or ""),
            "ranking":        int(profile.get("ranking") or 0),
            "easy_solved":    easy,
            "medium_solved":  medium,
            "hard_solved":    hard,
            "total_solved":   total,
            "badges":         badges,
            "leetcode_score": score,
            "profile_url":    f"https://leetcode.com/{username}",
            "error":          None,
        }
        log.info(
            "LeetCode done: E=%d M=%d H=%d score=%.1f",
            easy, medium, hard, score,
        )
        return result

    except requests.exceptions.RequestException as e:
        log.warning("LeetCode fetch error: %s", e)
        return _empty_result(str(e))


def _compute_score(easy: int, medium: int, hard: int) -> float:
    """
    Weighted score: easy*1 + medium*2 + hard*3, normalised 0-100.
    500-point raw = full score.
    """
    raw = float(easy) * 1.0 + float(medium) * 2.0 + float(hard) * 3.0
    return min(round(raw / 500.0 * 100.0, 1), 100.0)


def _empty_result(reason: str) -> Dict[str, Any]:
    return {
        "username":      None,
        "real_name":     None,
        "ranking":       0,
        "easy_solved":   0,
        "medium_solved": 0,
        "hard_solved":   0,
        "total_solved":  0,
        "badges":        [],
        "leetcode_score": 0.0,
        "profile_url":   None,
        "error":         reason,
    }

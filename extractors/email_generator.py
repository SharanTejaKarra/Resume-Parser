"""
extractors/email_generator.py  –  Auto Email Generator

Rules:
  - Uses ONE LLM call per email type, but only when triggered by the recruiter.
  - Falls back to a high-quality rule-based template if LLM is unavailable.
  - No LLM call happens automatically during resume processing (cost-safe).
"""

import os
from typing import Any, Dict, List, Optional
from utils.logger import get_logger

log = get_logger("email_generator")

# ── System prompt shared across email types ───────────────────────────────────

_EMAIL_SYSTEM = (
    "You are a professional HR recruiter writing emails on behalf of your company. "
    "Be concise, warm, and professional. Do NOT use placeholders like [NAME] — "
    "use the actual candidate name provided. Return ONLY the email body text "
    "(no subject line, no To/From headers). Plain text, 3–5 short paragraphs."
)


def _llm_generate(prompt: str) -> str:
    """Single LLM call returning raw email body text."""
    try:
        from extractors.llm_extractor import _chat
        result = _chat(prompt, system=_EMAIL_SYSTEM)
        return (result.get("content") or "").strip()
    except Exception as e:
        log.warning("LLM email generation failed: %s", e)
        return ""


# ── Shortlist email ───────────────────────────────────────────────────────────

def generate_shortlist_email(
    candidate: Dict[str, Any],
    role: str,
    ats_score: float,
    matched_skills: List[str],
    use_llm: bool = True,
) -> str:
    name   = candidate.get("name") or "Candidate"
    skills = ", ".join(matched_skills[:5]) or "your technical background"
    company = os.getenv("COMPANY_NAME", "our company")

    if use_llm:
        prompt = (
            f"Write a shortlist email for {name} who applied for the role of **{role}**.\n"
            f"Their ATS match score is {ats_score:.1f}/100.\n"
            f"Key matched skills: {skills}.\n"
            f"Company: {company}.\n"
            "Tone: professional, warm, positive. Mention their skills. "
            "Include a next steps paragraph (HR will contact to schedule screening call). "
            "Sign off with 'The Talent Acquisition Team'."
        )
        text = _llm_generate(prompt)
        if text:
            return text

    # Rule-based fallback
    return (
        f"Dear {name},\n\n"
        f"We are excited to inform you that after reviewing your application for the "
        f"**{role}** position at {company}, you have been shortlisted for the next stage "
        f"of our hiring process.\n\n"
        f"Your profile demonstrated strong alignment with our requirements, particularly "
        f"in areas such as {skills}. Your ATS compatibility score of {ats_score:.0f}/100 "
        "reflects a compelling match for this role.\n\n"
        "Our HR team will be in touch shortly to schedule a screening call. "
        "Please ensure your calendar is up to date and feel free to prepare any "
        "questions you may have about the role or our team.\n\n"
        "We look forward to connecting with you.\n\n"
        "Warm regards,\nThe Talent Acquisition Team"
    )


# ── Rejection email ───────────────────────────────────────────────────────────

def generate_rejection_email(
    candidate: Dict[str, Any],
    role: str,
    ats_score: float,
    missing_skills: List[str],
    use_llm: bool = True,
) -> str:
    name    = candidate.get("name") or "Candidate"
    gaps    = ", ".join(missing_skills[:3]) or "some key technical areas highlighted in our JD"
    company = os.getenv("COMPANY_NAME", "our company")

    if use_llm:
        prompt = (
            f"Write a polite rejection email for {name} who applied for **{role}** at {company}.\n"
            f"Their ATS score was {ats_score:.1f}/100.\n"
            f"Missing skills / gaps: {gaps}.\n"
            "Tone: polite, constructive, encouraging. Do NOT be harsh. "
            "Include 2-3 specific and actionable improvement suggestions. "
            "Encourage them to apply again in the future. "
            "Sign off with 'The Talent Acquisition Team'."
        )
        text = _llm_generate(prompt)
        if text:
            return text

    # Rule-based fallback
    suggestions = [f"Build deeper expertise in: {g}" for g in (missing_skills[:3] or ["core role requirements"])]
    sug_lines   = "\n".join(f"  • {s}" for s in suggestions)

    return (
        f"Dear {name},\n\n"
        f"Thank you for taking the time to apply for the **{role}** position at {company}. "
        "We genuinely appreciate your interest in joining our team and the effort you put into "
        "your application.\n\n"
        "After careful evaluation, we have decided to move forward with candidates whose "
        "profiles more closely align with the specific requirements of this role at this time. "
        "This decision was not easy given the calibre of applicants we received.\n\n"
        "To support your continued growth, here are a few areas that could strengthen future applications:\n"
        f"{sug_lines}\n\n"
        "We encourage you to keep building on your skills and to consider applying for future "
        "openings with us — your profile may be a great fit for other roles down the line.\n\n"
        "We wish you all the best in your career journey.\n\n"
        "Kind regards,\nThe Talent Acquisition Team"
    )


# ── Interview invite email ────────────────────────────────────────────────────

def generate_interview_email(
    candidate: Dict[str, Any],
    role: str,
    round_info: str = "Technical Round 1",
    use_llm: bool = True,
) -> str:
    name    = candidate.get("name") or "Candidate"
    company = os.getenv("COMPANY_NAME", "our company")

    if use_llm:
        prompt = (
            f"Write a professional interview invite email for {name} for the role of **{role}** at {company}.\n"
            f"Interview round: {round_info}.\n"
            "Include: brief intro, interview round details, "
            "ask them to confirm availability within 2 business days, "
            "mention they can reach out with any questions. "
            "Sign off with 'The Talent Acquisition Team'."
        )
        text = _llm_generate(prompt)
        if text:
            return text

    # Rule-based fallback
    return (
        f"Dear {name},\n\n"
        f"We are pleased to invite you to the next stage of our hiring process for the "
        f"**{role}** position at {company}.\n\n"
        f"You have been selected to participate in: **{round_info}**.\n\n"
        "This interview will give us an opportunity to dive deeper into your technical "
        "skills and for you to learn more about the role and our engineering culture.\n\n"
        "Please reply to this email within **2 business days** to confirm your availability, "
        "and share 2–3 preferred time slots (including your timezone). "
        "A member of our team will then send you a calendar invite with all the details.\n\n"
        "If you have any questions in the meantime, please don't hesitate to reach out.\n\n"
        "Looking forward to speaking with you!\n\n"
        "Best regards,\nThe Talent Acquisition Team"
    )


# ── Unified entry point ───────────────────────────────────────────────────────

def generate_all_emails(
    candidate: Dict[str, Any],
    role: str,
    ats_score: float,
    matched_skills: List[str],
    missing_skills: List[str],
    round_info: str = "Technical Round 1",
    use_llm: bool = True,
) -> Dict[str, str]:
    """
    Generate all 3 email types in a single call.
    Each email is generated independently (3 LLM calls max if use_llm=True).

    Returns:
    {
        "shortlist_email": "...",
        "rejection_email": "...",
        "interview_email": "..."
    }
    """
    log.info("Generating emails for '%s' | role=%s | use_llm=%s",
             candidate.get("name"), role, use_llm)

    return {
        "shortlist_email": generate_shortlist_email(
            candidate, role, ats_score, matched_skills, use_llm
        ),
        "rejection_email": generate_rejection_email(
            candidate, role, ats_score, missing_skills, use_llm
        ),
        "interview_email": generate_interview_email(
            candidate, role, round_info, use_llm
        ),
    }

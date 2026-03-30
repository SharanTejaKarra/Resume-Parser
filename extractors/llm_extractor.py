"""
extractors/llm_extractor.py  –  LLM-based semantic extraction
Supports two backends selected via LLM_PROVIDER env var:
  • "ollama"  – local Ollama or Ollama Cloud (default)
  • "openai"  – OpenAI GPT-3.5-turbo
Tracks every call through Langfuse.
"""
import json
import os
import re
import time
from typing import Any, Dict, Optional

from utils.logger import get_logger
from utils.langfuse_tracker import tracker

log = get_logger("llm_extractor")

# ── Active provider ────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower().strip()


# ── Ollama backend ─────────────────────────────────────────────────────────────

def _get_ollama_client():
    """Return a configured Ollama client (local or cloud)."""
    try:
        from ollama import Client
    except ImportError:
        raise ImportError("ollama package missing. Run: pip install ollama")

    host    = os.getenv("OLLAMA_HOST", "https://ollama.com")
    api_key = os.getenv("OLLAMA_API_KEY", "")

    headers: Dict[str, str] = {}
    if api_key and not api_key.startswith("your_"):
        headers["Authorization"] = f"Bearer {api_key}"
        log.info("Using Ollama Cloud at %s", host)
    else:
        host = "http://localhost:11434"
        log.info("Using local Ollama at %s", host)

    if headers:
        return Client(host=host, headers=headers)
    return Client(host=host)


def _chat_ollama(prompt: str, system: str = "", model: Optional[str] = None) -> Dict[str, Any]:
    """
    Raw Ollama chat call (ollama v0.6.x).
    Returns {"content": str, "input_tokens": int, "output_tokens": int, "model": str, "latency_ms": int}
    """
    model = model or os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    client = _get_ollama_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.time()
    try:
        response = client.chat(model=model, messages=messages, stream=False)
    except Exception as e:
        log.error("Ollama call failed: %s", e)
        raise

    latency_ms = int((time.time() - t0) * 1000)

    # ollama v0.6 returns a Pydantic ChatResponse object
    try:
        content = response.message.content  # type: ignore[union-attr]
    except AttributeError:
        content = str(response)

    try:
        input_tokens  = getattr(response, "prompt_eval_count", None) or len(prompt.split())
        output_tokens = getattr(response, "eval_count", None)         or len(content.split())
    except Exception:
        input_tokens  = len(prompt.split())
        output_tokens = len(content.split()) if content else 0

    log.info(
        "Ollama | model=%s latency=%dms in_tok=%d out_tok=%d",
        model, latency_ms, input_tokens, output_tokens,
    )
    return {
        "content":       content or "",
        "input_tokens":  int(input_tokens),
        "output_tokens": int(output_tokens),
        "model":         model,
        "latency_ms":    latency_ms,
    }


# ── OpenAI backend ─────────────────────────────────────────────────────────────

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


def _get_openai_client():
    """Return a configured OpenAI client."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package missing. Run: pip install openai"
        )
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )
    return OpenAI(api_key=api_key)


def _chat_openai(prompt: str, system: str = "", model: Optional[str] = None) -> Dict[str, Any]:
    """
    OpenAI chat-completion call (GPT-3.5-turbo by default).
    Returns the same shape as _chat_ollama so all callers are provider-agnostic.
    """
    model = model or OPENAI_MODEL
    client = _get_openai_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.2,
        )
    except Exception as e:
        log.error("OpenAI call failed: %s", e)
        raise

    latency_ms    = int((time.time() - t0) * 1000)
    content       = response.choices[0].message.content or ""
    input_tokens  = response.usage.prompt_tokens     if response.usage else len(prompt.split())
    output_tokens = response.usage.completion_tokens if response.usage else len(content.split())

    log.info(
        "OpenAI | model=%s latency=%dms in_tok=%d out_tok=%d",
        model, latency_ms, input_tokens, output_tokens,
    )
    return {
        "content":       content,
        "input_tokens":  int(input_tokens),
        "output_tokens": int(output_tokens),
        "model":         model,
        "latency_ms":    latency_ms,
    }


# ── Unified dispatcher ─────────────────────────────────────────────────────────

def _chat(prompt: str, system: str = "", model: Optional[str] = None) -> Dict[str, Any]:
    """
    Route to the active LLM backend (controlled by LLM_PROVIDER env var).
      LLM_PROVIDER=ollama  → Ollama (local or cloud)   [default]
      LLM_PROVIDER=openai  → OpenAI GPT-3.5-turbo
    """
    if LLM_PROVIDER == "openai":
        return _chat_openai(prompt, system=system, model=model)
    return _chat_ollama(prompt, system=system, model=model)


def _safe_parse_json(text: str) -> dict:
    """
    Extract valid JSON from LLM response (handles markdown fences, preamble).
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    cleaned = cleaned.rstrip("`").strip()

    # Find first complete { ... } block
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try parsing the whole cleaned string
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return {"raw_response": text}


# ─────────────────────────────────────────────────────────────────────────────
# Resume extraction
# ─────────────────────────────────────────────────────────────────────────────

RESUME_SYSTEM = (
    "You are an expert resume parser. "
    "Extract structured information accurately. "
    "IMPORTANT: Return ONLY valid JSON. Do NOT add any text before or after. "
    "Do NOT hallucinate — if information is missing, return null or []."
)

RESUME_PROMPT_TEMPLATE = """Extract the following fields from this resume text.
Return ONLY valid JSON (no markdown fences, no explanations).

CRITICAL EXPERIENCE RULES — READ CAREFULLY:
1. "full_time_experience_years" = ONLY paid full-time permanent jobs (NOT internships, NOT education, NOT projects, NOT certifications)
2. "internship_months" = total months across ALL internships only. Internships appear as "Intern", "Trainee", "Apprentice" in job titles
3. "is_student" = true if the resume shows an ONGOING bachelor's/master's degree (B.Tech, B.E., B.Sc, M.Tech etc.)
4. "candidate_type" = "student" | "fresher" | "experienced"
   - student: currently enrolled in undergrad/grad degree
   - fresher: recently graduated, no full-time work experience
   - experienced: has 1+ years of full-time work experience
5. NEVER add education dates (B.Tech 2022–2026) to experience
6. NEVER add project dates or hackathon dates to experience
7. A B.Tech/B.Sc student with only internships should have full_time_experience_years = 0

IMPORTANT SKILL INFERENCE RULES:
- If resume mentions LangGraph, LlamaIndex, or LangChain → add "LLM Orchestration"
- If resume mentions RAG, vector search, or embeddings → add "NLP", "Vector Database"
- If resume mentions FastAPI, Django, or Flask → add "Python", "REST API"
- If resume mentions K8s or kubernetes → add "Kubernetes"
- Extract ALL technologies from project descriptions, not just bullet points

JSON format:
{{
  "name": "full name or null",
  "summary": "professional summary or null",
  "skills": ["Python", "FastAPI", "LangChain"],
  "work_experience": [
    {{
      "company": "Acme Corp",
      "title": "Backend Intern",
      "is_internship": true,
      "start": "Jun 2024",
      "end": "Aug 2024",
      "duration_months": 3,
      "description": "Built API endpoints",
      "achievements": ["Reduced latency by 40%"]
    }}
  ],
  "projects": [
    {{
      "name": "AI Resume Parser",
      "description": "Built end-to-end AI pipeline",
      "tech_stack": ["Python", "FastAPI", "LangChain", "Streamlit"],
      "url": null
    }}
  ],
  "education": [
    {{
      "institution": "MIT",
      "degree": "B.Tech",
      "field": "Computer Science",
      "year": "2022–2026",
      "gpa": "3.8",
      "is_ongoing": true
    }}
  ],
  "certifications": ["AWS Solutions Architect"],
  "achievements": ["Winner of XYZ Hackathon"],
  "languages": ["English", "Hindi"],
  "full_time_experience_years": 0,
  "internship_months": 6,
  "is_student": true,
  "candidate_type": "student"
}}

If a field has no data, return null or empty list []. Do NOT fabricate data.

Resume Text:
{text}"""


def extract_resume_llm(text: str, candidate_name: str = "candidate") -> Dict[str, Any]:
    """
    Use LLM to semantically extract all resume fields.
    Returns {"data": dict, "langfuse": dict}.
    """
    # Guard against huge tokens
    truncated = text[:8000]
    prompt    = RESUME_PROMPT_TEMPLATE.format(text=truncated)

    log.info("LLM extraction for '%s' (chars=%d)", candidate_name, len(text))

    result = _chat(prompt, system=RESUME_SYSTEM)
    parsed = _safe_parse_json(result["content"])

    lf_summary = tracker.track_llm_call(
        trace_name      = "resume-parser",
        generation_name = "resume-extraction",
        model           = result["model"],
        prompt          = prompt,
        response        = result["content"],
        input_tokens    = result["input_tokens"],
        output_tokens   = result["output_tokens"],
        metadata        = {"candidate": candidate_name, "latency_ms": result["latency_ms"]},
    )

    return {"data": parsed, "langfuse": lf_summary}


# ─────────────────────────────────────────────────────────────────────────────
# JD extraction
# ─────────────────────────────────────────────────────────────────────────────

JD_SYSTEM = (
    "You are an expert job description analyzer. "
    "Return ONLY valid JSON. No text before or after the JSON object."
)

JD_PROMPT_TEMPLATE = """Analyze this job description and extract structured information.
Return ONLY valid JSON (no markdown fences):

{{
  "role": "exact job title",
  "company": "company name or null",
  "required_skills": ["Python", "Docker"],
  "preferred_skills": ["Kubernetes", "Terraform"],
  "min_experience_years": 3,
  "education_required": "B.Tech CS or equivalent",
  "responsibilities": ["Design scalable APIs", "Lead code reviews"],
  "industry": "FinTech / AI / SaaS etc.",
  "employment_type": "Full-time",
  "key_technologies": ["FastAPI", "PostgreSQL", "Redis"]
}}

IMPORTANT: Extract ALL skills mentioned anywhere in the JD (job title, requirements,
responsibilities, nice-to-haves). Be thorough — missing a skill = bad match score.

Job Description:
{text}"""


def extract_jd_llm(text: str) -> Dict[str, Any]:
    """Extract structured fields from a job description."""
    truncated = text[:6000]
    prompt    = JD_PROMPT_TEMPLATE.format(text=truncated)

    log.info("LLM JD extraction (chars=%d)", len(text))

    result = _chat(prompt, system=JD_SYSTEM)
    parsed = _safe_parse_json(result["content"])

    lf_summary = tracker.track_llm_call(
        trace_name      = "resume-parser",
        generation_name = "jd-extraction",
        model           = result["model"],
        prompt          = prompt,
        response        = result["content"],
        input_tokens    = result["input_tokens"],
        output_tokens   = result["output_tokens"],
        metadata        = {"step": "jd_parsing", "latency_ms": result["latency_ms"]},
    )

    return {"data": parsed, "langfuse": lf_summary}


# ─────────────────────────────────────────────────────────────────────────────
# Comparison generation
# ─────────────────────────────────────────────────────────────────────────────

COMPARE_SYSTEM = "You are an expert technical recruiter. Be concise and data-driven."

COMPARE_PROMPT_TEMPLATE = """Compare these candidates for the following job and explain their ranking.

Job Description Summary:
{jd_summary}

Candidates (ranked by ATS score, highest first):
{candidates_summary}

Provide:
1. A comparison paragraph (3-5 sentences) explaining why the top candidate ranked highest.
2. Top 3 strengths of the #1 ranked candidate.
3. Key gaps the lower-ranked candidates need to address.

Be specific — mention actual skills, projects, and scores."""


def generate_comparison_explanation(
    jd_data: dict,
    candidates: list,
    ranked_results: list,
) -> Dict[str, Any]:
    """Generate LLM explanation for candidate ranking."""
    role       = jd_data.get("role", "the role")
    req_skills = ", ".join(list(jd_data.get("required_skills") or [])[:10])
    jd_summary = f"Role: {role}\nRequired Skills: {req_skills}"

    lines = []
    for i, r in enumerate(ranked_results[:5]):
        name   = r.get("name", f"Candidate {i+1}")
        score  = r.get("ats_score", 0)
        skills = ", ".join(list(r.get("skills") or [])[:6])
        exp    = r.get("total_experience_years", 0)
        lines.append(
            f"#{i+1} {name} | ATS={score:.1f} | Exp={exp}y | Skills: {skills}"
        )

    prompt = COMPARE_PROMPT_TEMPLATE.format(
        jd_summary         = jd_summary,
        candidates_summary = "\n".join(lines),
    )

    result = _chat(prompt, system=COMPARE_SYSTEM)

    lf_summary = tracker.track_llm_call(
        trace_name      = "resume-parser",
        generation_name = "candidate-comparison",
        model           = result["model"],
        prompt          = prompt,
        response        = result["content"],
        input_tokens    = result["input_tokens"],
        output_tokens   = result["output_tokens"],
        metadata        = {
            "num_candidates": len(ranked_results),
            "latency_ms":     result["latency_ms"],
        },
    )

    return {"explanation": result["content"], "langfuse": lf_summary}

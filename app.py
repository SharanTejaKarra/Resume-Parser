"""
app.py  –  AI Resume Parser  |  Column-aware · LLM · Embeddings · ATS · Langfuse
"""
# ── stdlib ────────────────────────────────────────────────────────────────────
import io
import os
import json
import time
import traceback
from typing import Dict, Any, List, Optional

# ── env ───────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Streamlit ─────────────────────────────────────────────────────────────────
import streamlit as st

# ── internal ──────────────────────────────────────────────────────────────────
from utils.logger import get_logger
from utils.langfuse_tracker import tracker as lf_tracker

from parsers.pdf_parser  import extract_pdf, extract_pdf_pdfplumber_fallback
from parsers.docx_parser import extract_docx

from extractors.regex_extractor import extract_regex_fields
from extractors.llm_extractor   import (
    extract_resume_llm,
    extract_jd_llm,
    generate_comparison_explanation,
)

from analyzers.github_analyzer   import analyze_github
from analyzers.leetcode_analyzer import analyze_leetcode

from scoring.embedding_matcher import compute_jd_similarity, compute_skill_match
from scoring.ats_scorer        import compute_ats_score

log = get_logger("app")

# ─────────────────────────────────────────────────────────────────────────────
# ██████  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Parser",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# ██████  CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root palette ───────────────────────────────────── */
:root {
  --bg:        #0d0f16;
  --surface:   #151823;
  --card:      #1c2033;
  --border:    #2a2f45;
  --accent:    #6c63ff;
  --accent2:   #00d2ff;
  --accent3:   #ff6b9d;
  --success:   #00e5a0;
  --warn:      #ffb340;
  --text:      #e8eaf0;
  --subtext:   #8890a6;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--text) !important;
}

/* ── Sidebar ────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}

/* ── Metric cards ───────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 18px;
  transition: transform .2s, box-shadow .2s;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(108,99,255,.25);
}
[data-testid="stMetricLabel"]  { color: var(--subtext) !important; font-size: 12px; }
[data-testid="stMetricValue"]  { color: var(--text) !important;    font-size: 28px; font-weight: 700; }
[data-testid="stMetricDelta"]  { font-size: 12px; }

/* ── Tabs ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: var(--surface);
  border-radius: 12px;
  padding: 4px;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  color: var(--subtext);
  border-radius: 8px;
  font-weight: 500;
  transition: all .2s;
}
.stTabs [aria-selected="true"] {
  background: var(--accent) !important;
  color: #fff !important;
}

/* ── Buttons ────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: #fff;
  border: none;
  border-radius: 10px;
  font-weight: 600;
  padding: 10px 24px;
  transition: opacity .2s, transform .15s;
}
.stButton > button:hover {
  opacity: .88;
  transform: scale(1.02);
}

/* ── File uploader ──────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
  background: var(--card) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 12px !important;
  transition: border-color .3s;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--accent) !important;
}

/* ── Text inputs ────────────────────────────────────── */
textarea, input[type="text"], .stTextArea textarea {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
}

/* ── Expander ───────────────────────────────────────── */
.streamlit-expanderHeader {
  background: var(--card) !important;
  border-radius: 10px !important;
  font-weight: 600;
}

/* ── Code / JSON ────────────────────────────────────── */
.stCodeBlock, code {
  background: var(--card) !important;
  border: 1px solid var(--border);
  border-radius: 8px;
}

/* ── Progress bars ──────────────────────────────────── */
.stProgress > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
  border-radius: 999px;
}

/* ── Custom card component ──────────────────────────── */
.glass-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 22px 26px;
  margin-bottom: 18px;
  transition: transform .2s, box-shadow .2s;
}
.glass-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 30px rgba(0,0,0,.4);
}

/* ── Hero banner ────────────────────────────────────── */
.hero-banner {
  background: linear-gradient(135deg, #1a1040 0%, #0d1836 50%, #0d2040 100%);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 40px 48px;
  margin-bottom: 32px;
  position: relative;
  overflow: hidden;
}
.hero-banner::before {
  content: '';
  position: absolute;
  top: -50%; left: -50%;
  width: 200%; height: 200%;
  background: radial-gradient(circle at 30% 50%, rgba(108,99,255,.15) 0%, transparent 60%),
              radial-gradient(circle at 70% 30%, rgba(0,210,255,.10) 0%, transparent 60%);
  pointer-events: none;
}
.hero-title {
  font-size: 2.8em;
  font-weight: 800;
  background: linear-gradient(135deg, #fff 30%, var(--accent2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0 0 8px;
}
.hero-sub {
  color: var(--subtext);
  font-size: 1.05em;
  margin: 0;
}

/* ── Rank badge ─────────────────────────────────────── */
.rank-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 14px;
  border-radius: 999px;
  font-weight: 700;
  font-size: .85em;
}
.rank-1 { background: linear-gradient(135deg,#ffd700,#ff9f00); color: #000; }
.rank-2 { background: linear-gradient(135deg,#c0c0c0,#8a8a8a); color: #000; }
.rank-3 { background: linear-gradient(135deg,#cd7f32,#8b4513); color: #fff; }
.rank-n { background: var(--border); color: var(--subtext); }

/* ── Skill pill ─────────────────────────────────────── */
.skill-pill {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 999px;
  font-size: .78em;
  font-weight: 500;
  margin: 2px;
}
.skill-match   { background: rgba(0,229,160,.15); color: var(--success); border:1px solid rgba(0,229,160,.3); }
.skill-missing { background: rgba(255,107,157,.12); color: var(--accent3); border:1px solid rgba(255,107,157,.3); }
.skill-neutral { background: rgba(108,99,255,.15); color: var(--accent); border:1px solid rgba(108,99,255,.3); }

/* ── Score ring helper ──────────────────────────────── */
.score-big {
  font-size: 3.5em;
  font-weight: 800;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

/* ── Langfuse log table ─────────────────────────────── */
.lf-log-item {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 16px;
  margin: 6px 0;
  font-size: .85em;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ██████  Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "jd_data":          None,
        "jd_text":          "",
        "candidates":       [],   # list of processed candidate dicts
        "ranked":           [],   # sorted by ATS score
        "comparison":       None,
        "lf_logs":          [],   # Langfuse tracking records
        "processing_log":   [],   # step-by-step human log
        "total_tokens":     0,
        "total_cost":       0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# ██████  Helper utilities
# ─────────────────────────────────────────────────────────────────────────────
def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.processing_log.append(f"[{ts}] {msg}")
    log.info(msg)


def _add_lf(record: dict):
    if record:
        st.session_state.lf_logs.append(record)
        st.session_state.total_tokens += record.get("total_tokens", 0)
        st.session_state.total_cost   += record.get("cost_usd", 0.0)


def _score_color(score: float) -> str:
    if score >= 80: return "#00e5a0"
    if score >= 60: return "#ffb340"
    return "#ff6b9d"


def _parse_file(uploaded) -> Optional[Dict]:
    """Multi-stage parse: tries PyMuPDF → pdfplumber → pdfminer automatically."""
    raw = uploaded.read()
    name = uploaded.name.lower()
    try:
        if name.endswith(".pdf"):
            result = extract_pdf(raw)   # handles all 3 stages internally
        elif name.endswith(".docx"):
            result = extract_docx(raw)
        else:
            st.error(f"Unsupported format: {uploaded.name}")
            return None
        return result
    except Exception as e:
        st.error(f"Parse error ({uploaded.name}): {e}")
        log.error(traceback.format_exc())
        return None


def _safe_list(val) -> list:
    if isinstance(val, list): return val
    if isinstance(val, str) and val: return [val]
    return []


def _build_candidate(name: str, parse_result: dict, llm_data: dict, regex_data: dict,
                     gh: dict, lc: dict, jd_sim: float, skill_pct: float,
                     matched_skills: list, missing_skills: list, ats: dict) -> dict:
    """Merge all extracted data into a single candidate dict."""
    # ── Resolve experience from LLM output (preferred) or regex fallback ───────
    llm_ft     = llm_data.get("full_time_experience_years")   # new field
    llm_intern = llm_data.get("internship_months")            # new field
    llm_student= llm_data.get("is_student")                   # new field
    llm_ctype  = llm_data.get("candidate_type")               # new field

    ft_years      = float(llm_ft    if llm_ft    is not None else regex_data.get("full_time_experience_years", 0))
    i_months      = float(llm_intern if llm_intern is not None else regex_data.get("internship_months", 0))
    is_student    = bool(llm_student if llm_student is not None else regex_data.get("is_student", False))
    ctype         = llm_ctype or regex_data.get("candidate_type", "fresher")

    intern_w      = round(i_months / 12.0 * 0.3, 2)
    effective_exp = round(ft_years + intern_w, 2)

    return {
        "name":               llm_data.get("name") or name,
        "email":              regex_data.get("email"),
        "phone":              regex_data.get("phone"),
        "github_url":         regex_data.get("github_url"),
        "linkedin_url":       regex_data.get("linkedin_url"),
        "leetcode_url":       regex_data.get("leetcode_url"),
        "summary":            llm_data.get("summary", ""),
        "skills":             _safe_list(llm_data.get("skills")) or regex_data.get("tech_skills", []),
        "work_experience":    _safe_list(llm_data.get("work_experience")),
        "projects":           _safe_list(llm_data.get("projects")),
        "education":          _safe_list(llm_data.get("education")),
        "certifications":     _safe_list(llm_data.get("certifications")),
        "achievements":       _safe_list(llm_data.get("achievements")),
        # ── experience breakdown ───────────────────────────────────────────────
        "full_time_exp_years":    ft_years,
        "internship_months":      i_months,
        "effective_exp_years":    effective_exp,
        "is_student":             is_student,
        "candidate_type":         ctype,
        # legacy alias kept for display
        "total_experience_years": effective_exp,
        # ── scoring ───────────────────────────────────────────────────────────
        "github":             gh,
        "leetcode":           lc,
        "jd_similarity":      jd_sim,
        "skill_match_pct":    skill_pct,
        "matched_skills":     matched_skills,
        "missing_skills":     missing_skills,
        "ats_score":          ats["ats_score"],
        "ats_breakdown":      ats,
        "is_two_column":      parse_result.get("is_two_column", False),
        "pages":              parse_result.get("pages", 1),
        "parse_status":       parse_result.get("parse_status", "OK"),
        "parser_used":        parse_result.get("parser_used", "unknown"),
        "char_count":         parse_result.get("char_count", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ██████  Sidebar  –  stats only, config → .env
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 0 4px;">
      <div style="font-size:2em;">🧠</div>
      <div style="font-weight:700; font-size:1.1em; color:#e8eaf0;">AI Resume Parser</div>
      <div style="font-size:.75em; color:#8890a6; margin-top:2px;">v2.0 · Student-aware</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Active config (read-only) ───────────────────────────────────────────
    st.markdown("### ⚙️ Active Config")
    _ol_model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    _ol_host  = os.getenv("OLLAMA_HOST",  "localhost:11434")
    _has_lf   = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
    _has_gh   = bool(os.getenv("GITHUB_TOKEN"))

    st.markdown(f"""
| Setting | Value |
|---|---|
| 🤖 **Model** | `{_ol_model}` |
| 🌐 **Host** | `{_ol_host}` |
| 🔭 **Langfuse** | {'✅ connected' if _has_lf else '⚠️ not set'} |
| 🐙 **GitHub** | {'✅ set' if _has_gh else '(unauth)'} |
    """)
    st.caption("📝 Edit `.env` and restart to change settings.")

    st.divider()

    # ── Session stats ──────────────────────────────────────────────────────
    st.markdown("### 📊 Session Stats")
    st.metric("🔤 Tokens",     f"{st.session_state.total_tokens:,}")
    st.metric("💰 Est. Cost",  f"${st.session_state.total_cost:.4f}")
    st.metric("📝 LLM Calls", len(st.session_state.lf_logs))
    st.metric("👥 Candidates", len(st.session_state.candidates))

    st.divider()

    # ── Enrichment toggles ─────────────────────────────────────────────────
    st.markdown("### 🔬 Enrichment")
    _enrich_gh = st.toggle("🐙 GitHub",   value=True, key="sidebar_gh")
    _enrich_lc = st.toggle("💡 LeetCode", value=True, key="sidebar_lc")

    st.divider()

    if st.button("🗑️ Reset Session", use_container_width=True):
        for k in ["jd_data","jd_text","candidates","ranked","comparison",
                  "lf_logs","processing_log","total_tokens","total_cost"]:
            st.session_state[k] = [] if isinstance(st.session_state[k], list) else (
                0.0 if isinstance(st.session_state[k], float) else
                0   if isinstance(st.session_state[k], int)   else None
            )
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ██████  Hero banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🧠 AI Resume Parser</div>
  <p class="hero-sub">
    Column-aware PDF parsing · LLM semantic extraction · Embedding job matching ·
    GitHub & LeetCode enrichment · ATS scoring · Langfuse observability
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ██████  Main tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_jd, tab_upload, tab_rank, tab_compare, tab_obs, tab_logs = st.tabs([
    "📋 Job Description",
    "📄 Upload Resumes",
    "🏆 Rankings",
    "⚖️  Compare",
    "🔭 Observability",
    "📜 Logs",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Job Description
# ═══════════════════════════════════════════════════════════════════════════════
with tab_jd:
    st.markdown("## 📋 Step 1 — Paste Job Description")
    st.markdown("The system will extract required skills, experience, and role details.")

    jd_input = st.text_area(
        "Job Description",
        height=300,
        placeholder="Paste the full job description here…",
        value=st.session_state.jd_text,
        key="jd_textarea",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        parse_jd = st.button("🚀 Analyse JD", use_container_width=True)

    if parse_jd and jd_input.strip():
        st.session_state.jd_text = jd_input
        with st.spinner("Parsing job description with LLM…"):
            try:
                jd_result = extract_jd_llm(jd_input)
                st.session_state.jd_data = jd_result["data"]
                _add_lf(jd_result["langfuse"])
                _log("✅ JD parsed successfully")
                st.success("Job description analysed!")
            except Exception as e:
                st.error(f"JD parsing failed: {e}")
                log.error(traceback.format_exc())

    if st.session_state.jd_data:
        jd = st.session_state.jd_data
        st.markdown("---")
        st.markdown("### 🎯 Extracted JD Fields")

        c1, c2, c3 = st.columns(3)
        c1.metric("Role",         jd.get("role", "—"))
        c2.metric("Min. Experience", f"{jd.get('min_experience_years', 0)} yrs")
        c3.metric("Employment",   jd.get("employment_type", "—"))

        col_req, col_pref = st.columns(2)
        with col_req:
            st.markdown("**Required Skills**")
            for s in _safe_list(jd.get("required_skills")):
                st.markdown(f'<span class="skill-pill skill-neutral">⚡ {s}</span>', unsafe_allow_html=True)

        with col_pref:
            st.markdown("**Preferred Skills**")
            for s in _safe_list(jd.get("preferred_skills")):
                st.markdown(f'<span class="skill-pill skill-match">✨ {s}</span>', unsafe_allow_html=True)

        with st.expander("📄 Full JD JSON"):
            st.json(jd)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Upload & Process Resumes
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("## 📄 Step 2 — Upload Resumes")

    if not st.session_state.jd_data:
        st.warning("⚠️  Please analyse a Job Description first (Tab 1).")
    else:
        uploaded_files = st.file_uploader(
            "Drop PDF or DOCX resumes here (multiple supported)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="resume_uploader",
        )

        enable_gh = st.session_state.get("sidebar_gh", True)
        enable_lc = st.session_state.get("sidebar_lc", True)

        if uploaded_files:
            if st.button("⚡ Process All Resumes", use_container_width=True):
                jd     = st.session_state.jd_data
                jd_txt = st.session_state.jd_text

                new_candidates = []
                prog = st.progress(0)
                status_box = st.empty()

                for idx, uf in enumerate(uploaded_files):
                    prog.progress((idx) / len(uploaded_files))
                    status_box.info(f"Processing **{uf.name}** ({idx+1}/{len(uploaded_files)})…")
                    _log(f"▶ Starting: {uf.name}")

                    try:
                        # ── 1. Parse document (multi-stage) ───────────────────────
                        parse_res = _parse_file(uf)
                        if not parse_res:
                            continue

                        parse_status = parse_res.get("parse_status", "OK")
                        parser_used  = parse_res.get("parser_used", "unknown")
                        char_count   = parse_res.get("char_count", 0)
                        full_text    = parse_res["full_text"]
                        col_type     = "Two-column" if parse_res["is_two_column"] else "Single-column"

                        _log(
                            f"  📄 Parsed ({col_type}, {parse_res['pages']} pages, "
                            f"{char_count} chars) via {parser_used} [{parse_status}]"
                        )

                        # ── Hard bail-out if all parsers returned <200 chars ───────
                        if parse_status == "PARSE_FAILED":
                            st.warning(
                                f"⚠️  **{uf.name}** — all PDF parsers returned insufficient text "
                                f"({char_count} chars). Skipping this resume. "
                                "Try re-saving the PDF as text-based (not image/scanned)."
                            )
                            _log(f"  ⚠️  PARSE_FAILED – skipping {uf.name}")
                            continue

                        if parse_status == "LOW_CONFIDENCE":
                            st.warning(
                                f"⚠️  **{uf.name}** — low text confidence ({char_count} chars). "
                                "Results may be inaccurate."
                            )
                            _log(f"  ⚠️  LOW_CONFIDENCE ({char_count} chars) for {uf.name}")

                        # ── 2. Regex extraction ────────────────────────────────────
                        regex = extract_regex_fields(full_text)
                        _log(
                            f"  🔍 Regex: email={regex.get('email')}, "
                            f"github={regex.get('github_username')}, "
                            f"skills={len(regex.get('tech_skills', []))}"
                        )

                        # ── 3. LLM extraction ──────────────────────────────────────
                        cand_label = uf.name.replace(".pdf","").replace(".docx","")
                        llm_res    = extract_resume_llm(full_text, candidate_name=cand_label)
                        llm_data   = llm_res["data"]
                        _add_lf(llm_res["langfuse"])
                        _log(f"  🤖 LLM extraction done (tokens={llm_res['langfuse'].get('total_tokens',0)})")

                        # ── 4. GitHub analysis ─────────────────────────────────────
                        gh: Dict[str, Any] = {"github_score": 0.0, "error": "disabled"}
                        if enable_gh and regex.get("github_username"):
                            gh = analyze_github(regex["github_username"])
                            _log(f"  🐙 GitHub: repos={gh.get('public_repos',0)} stars={gh.get('total_stars',0)}")

                        # ── 5. LeetCode analysis ───────────────────────────────────
                        lc: Dict[str, Any] = {"leetcode_score": 0.0, "error": "disabled"}
                        if enable_lc and regex.get("leetcode_username"):
                            lc = analyze_leetcode(regex["leetcode_username"])
                            _log(f"  💡 LeetCode: solved={lc.get('total_solved',0)}")

                        # ── 6. Embedding JD similarity ─────────────────────────────
                        jd_sim = compute_jd_similarity(full_text, jd_txt)
                        _log(f"  📐 JD similarity: {jd_sim:.4f}")

                        # ── 7. Skill match (LLM skills + regex skills merged) ──────
                        all_skills = sorted(set(
                            _safe_list(llm_data.get("skills")) +
                            regex.get("tech_skills", [])
                        ))
                        jd_req_skills = _safe_list(jd.get("required_skills"))
                        skill_pct, matched, missing = compute_skill_match(all_skills, jd_req_skills)
                        _log(f"  🎯 Skill match: {skill_pct:.1f}% ({len(matched)}/{len(jd_req_skills)} required)")

                        # ── 8. ATS score (new signature) ───────────────────────────
                        # Resolve experience from LLM first, fallback to regex
                        ft_exp = float(
                            llm_data.get("full_time_experience_years")
                            if llm_data.get("full_time_experience_years") is not None
                            else regex.get("full_time_experience_years", 0)
                        )
                        i_months = float(
                            llm_data.get("internship_months")
                            if llm_data.get("internship_months") is not None
                            else regex.get("internship_months", 0)
                        )
                        is_stud = bool(
                            llm_data.get("is_student")
                            if llm_data.get("is_student") is not None
                            else regex.get("is_student", False)
                        )
                        ctype = llm_data.get("candidate_type") or regex.get("candidate_type", "fresher")

                        _log(
                            f"  💼 Experience: type={ctype} ft={ft_exp}yr "
                            f"intern={i_months:.0f}mo student={is_stud}"
                        )

                        ats = compute_ats_score(
                            jd_similarity         = jd_sim,
                            skill_match_pct       = skill_pct,
                            full_time_exp_years   = ft_exp,
                            internship_months     = i_months,
                            is_student            = is_stud,
                            candidate_type        = ctype,
                            min_exp_required      = float(jd.get("min_experience_years", 2)),
                            github_score          = float(gh.get("github_score", 0)),
                            leetcode_score        = float(lc.get("leetcode_score", 0)),
                            projects              = _safe_list(llm_data.get("projects")),
                        )
                        _log(
                            f"  ✅ ATS score: {ats['ats_score']} "
                            f"(mode={ats['scoring_mode']})"
                        )

                        # ── 9. Build candidate ─────────────────────────────────────
                        candidate = _build_candidate(
                            name=cand_label, parse_result=parse_res,
                            llm_data=llm_data, regex_data=regex,
                            gh=gh, lc=lc,
                            jd_sim=jd_sim, skill_pct=skill_pct,
                            matched_skills=matched, missing_skills=missing,
                            ats=ats,
                        )
                        new_candidates.append(candidate)
                        _log(f"  🏁 Done → ATS={ats['ats_score']} parser={parser_used}")

                    except Exception as e:
                        st.error(f"Error processing {uf.name}: {e}")
                        log.error(traceback.format_exc())
                        _log(f"  ❌ Error: {e}")

                prog.progress(1.0)
                status_box.empty()

                # merge with existing
                existing_names = {c["name"] for c in st.session_state.candidates}
                for c in new_candidates:
                    if c["name"] not in existing_names:
                        st.session_state.candidates.append(c)

                # rank
                st.session_state.ranked = sorted(
                    st.session_state.candidates,
                    key=lambda c: c["ats_score"],
                    reverse=True,
                )
                st.session_state.comparison = None  # reset comparison
                st.success(f"✅ Processed {len(new_candidates)} resume(s)!")
                st.balloons()

        # ── Show already-processed candidates ──────────────────────────────────
        if st.session_state.candidates:
            st.markdown("---")
            st.markdown(f"### 👥 Processed Candidates ({len(st.session_state.candidates)})")
            for c in st.session_state.candidates:
                parse_badge = {
                    "OK":             "🟢 OK",
                    "LOW_CONFIDENCE": "🟡 Low Confidence",
                    "PARSE_FAILED":   "🔴 Failed",
                }.get(c.get("parse_status", "OK"), "🟢 OK")

                ctype_emoji = {"student": "🎓", "fresher": "🌱", "experienced": "💼"}.get(
                    c.get("candidate_type", "fresher"), "💼"
                )

                with st.expander(
                    f"**{c['name']}**  |  ATS: {c['ats_score']}  |  "
                    f"{ctype_emoji} {c.get('candidate_type','—').capitalize()}  |  "
                    f"Parse: {parse_badge}"
                ):
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("ATS Score",     f"{c['ats_score']}")
                    col_b.metric("JD Similarity", f"{c['jd_similarity']:.2f}")
                    col_c.metric("Skill Match",   f"{c['skill_match_pct']:.1f}%")
                    col_d.metric("Scoring Mode",  c.get('ats_breakdown', {}).get('scoring_mode', '—').capitalize())

                    # experience detail row
                    ec1, ec2, ec3 = st.columns(3)
                    ec1.metric("Full-time exp",   f"{c.get('full_time_exp_years', 0):.1f} yrs")
                    ec2.metric("Internship",       f"{c.get('internship_months', 0):.0f} months")
                    ec3.metric("Effective exp",    f"{c.get('effective_exp_years', 0):.2f} yrs")

                    pcol1, pcol2, pcol3 = st.columns(3)
                    pcol1.markdown(f"**Parser:** `{c.get('parser_used','—')}`")
                    pcol2.markdown(f"**Layout:** {'Two-column 📐' if c['is_two_column'] else 'Single-column 📄'}")
                    pcol3.markdown(f"**Text chars:** `{c.get('char_count', 0):,}`")

                    st.markdown("**Matched Skills:**")
                    for s in c["matched_skills"]:
                        st.markdown(f'<span class="skill-pill skill-match">✓ {s}</span>', unsafe_allow_html=True)

                    st.markdown("**Missing Skills:**")
                    for s in c["missing_skills"]:
                        st.markdown(f'<span class="skill-pill skill-missing">✗ {s}</span>', unsafe_allow_html=True)

                    if c["github"].get("public_repos"):
                        gh_d = c["github"]
                        st.markdown(
                            f"**GitHub:** [{gh_d['username']}]({gh_d['profile_url']}) · "
                            f"{gh_d['public_repos']} repos · ⭐ {gh_d['total_stars']} · "
                            f"Score: **{gh_d['github_score']}**"
                        )
                    if c["leetcode"].get("total_solved"):
                        lc_d = c["leetcode"]
                        st.markdown(
                            f"**LeetCode:** [{lc_d['username']}]({lc_d['profile_url']}) · "
                            f"🟢 {lc_d['easy_solved']} 🟡 {lc_d['medium_solved']} 🔴 {lc_d['hard_solved']} · "
                            f"Score: **{lc_d['leetcode_score']}**"
                        )

                    with st.expander("🔍 Full Extracted Data"):
                        clean = {k: v for k, v in c.items()
                                 if k not in ["github", "leetcode", "ats_breakdown"]}
                        st.json(clean)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Rankings
# ═══════════════════════════════════════════════════════════════════════════════
with tab_rank:
    st.markdown("## 🏆 Candidate Rankings")

    if not st.session_state.ranked:
        st.info("Process at least one resume to see rankings.")
    else:
        ranked = st.session_state.ranked
        import plotly.graph_objects as go
        import plotly.express as px

        # ── Bar chart ───────────────────────────────────────────────────────────
        names  = [c["name"] for c in ranked]
        scores = [c["ats_score"] for c in ranked]
        colors = [_score_color(s) for s in scores]

        fig = go.Figure(go.Bar(
            x=names, y=scores,
            marker=dict(
                color=scores,
                colorscale=[[0,"#ff6b9d"],[0.5,"#ffb340"],[1,"#00e5a0"]],
                cmin=0, cmax=100,
                showscale=True,
                colorbar=dict(title="ATS"),
            ),
            text=[f"{s}" for s in scores],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>ATS: %{y}<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d0f16",
            plot_bgcolor="#151823",
            title=dict(text="ATS Score Ranking", font=dict(size=18)),
            yaxis=dict(range=[0, 110], gridcolor="#2a2f45"),
            xaxis=dict(gridcolor="#2a2f45"),
            margin=dict(t=60, b=20, l=20, r=20),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Radar chart for top 3 ───────────────────────────────────────────────
        if len(ranked) >= 2:
            st.markdown("### 🕸️ Skills Radar — Top 3")
            cats = ["JD Similarity×100", "Skill Match", "Exp Score", "GitHub", "LeetCode", "Projects"]
            radar_fig = go.Figure()
            for c in ranked[:3]:
                bd = c["ats_breakdown"]
                vals = [
                    c["jd_similarity"] * 100,
                    c["skill_match_pct"],
                    bd["experience_score"],
                    c["github"].get("github_score", 0),
                    c["leetcode"].get("leetcode_score", 0),
                    bd["project_score"],
                ]
                radar_fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=cats + [cats[0]],
                    fill="toself",
                    name=c["name"],
                    opacity=0.75,
                ))
            radar_fig.update_layout(
                polar=dict(
                    bgcolor="#151823",
                    radialaxis=dict(visible=True, range=[0,100], gridcolor="#2a2f45"),
                    angularaxis=dict(gridcolor="#2a2f45"),
                ),
                paper_bgcolor="#0d0f16",
                template="plotly_dark",
                height=420,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        # ── Rank table ──────────────────────────────────────────────────────────
        st.markdown("### 📋 Detailed Ranking Table")
        rank_badges = ["🥇","🥈","🥉"] + ["🎖️"] * 20

        for i, c in enumerate(ranked):
            badge = rank_badges[i]
            color = _score_color(c["ats_score"])
            with st.container():
                st.markdown(f"""
<div class="glass-card">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <span style="font-size:1.5em">{badge}</span>
      <strong style="font-size:1.15em; margin-left:10px;">{c['name']}</strong>
      <span style="color:{color}; font-size:1.8em; font-weight:800; margin-left:20px;">{c['ats_score']}</span>
    </div>
    <div style="color:var(--subtext); font-size:.9em;">
      {c['email'] or '—'} &nbsp;|&nbsp; {c['total_experience_years']} yrs exp
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("JD Sim", f"{c['jd_similarity']:.3f}")
                c2.metric("Skill", f"{c['skill_match_pct']:.1f}%")
                c3.metric("GitHub", f"{c['github'].get('github_score',0):.1f}")
                c4.metric("LeetCode", f"{c['leetcode'].get('leetcode_score',0):.1f}")
                c5.metric("Projects", f"{c['ats_breakdown']['project_score']:.1f}")
                st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Compare
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("## ⚖️ Candidate Comparison")

    if len(st.session_state.candidates) < 2:
        st.info("Upload at least 2 resumes to compare.")
    else:
        names = [c["name"] for c in st.session_state.candidates]

        with st.form("compare_form"):
            c1_name = st.selectbox("Candidate A", names, index=0)
            c2_name = st.selectbox("Candidate B", names, index=min(1, len(names)-1))
            gen_exp  = st.checkbox("Generate LLM explanation", value=True)
            submitted = st.form_submit_button("⚖️ Compare")

        if submitted:
            ca = next(c for c in st.session_state.candidates if c["name"] == c1_name)
            cb = next(c for c in st.session_state.candidates if c["name"] == c2_name)

            st.markdown("---")
            col_a, col_b = st.columns(2)

            def _cand_card(col, cand, label):
                with col:
                    score_color = _score_color(cand["ats_score"])
                    st.markdown(f"""
<div class="glass-card">
  <div style="text-align:center;">
    <div style="font-weight:700; font-size:1.1em;">{label}</div>
    <div style="font-size:.9em; color:var(--subtext);">{cand['name']}</div>
    <div class="score-big" style="margin:8px 0;">{cand['ats_score']}</div>
    <div style="color:var(--subtext); font-size:.8em;">ATS Score</div>
  </div>
</div>
""", unsafe_allow_html=True)
                    st.metric("JD Similarity",  f"{cand['jd_similarity']:.3f}")
                    st.metric("Skill Match",    f"{cand['skill_match_pct']:.1f}%")
                    st.metric("Experience",     f"{cand['total_experience_years']} yrs")
                    st.metric("GitHub Score",   cand["github"].get("github_score", 0))
                    st.metric("LeetCode Score", cand["leetcode"].get("leetcode_score", 0))

                    st.markdown("**Skills on Resume:**")
                    for s in cand["skills"][:12]:
                        in_match = s in cand["matched_skills"]
                        cls = "skill-match" if in_match else "skill-neutral"
                        st.markdown(f'<span class="skill-pill {cls}">{s}</span>', unsafe_allow_html=True)

                    st.markdown("**Missing JD Skills:**")
                    for s in cand["missing_skills"][:8]:
                        st.markdown(f'<span class="skill-pill skill-missing">✗ {s}</span>', unsafe_allow_html=True)

                    if cand["work_experience"]:
                        st.markdown("**Experience:**")
                        for ex in cand["work_experience"][:3]:
                            if isinstance(ex, dict):
                                st.markdown(f"- **{ex.get('title','')}** @ {ex.get('company','')} ({ex.get('start','')}–{ex.get('end','')})")

                    if cand["projects"]:
                        st.markdown("**Projects:**")
                        for p in cand["projects"][:3]:
                            if isinstance(p, dict):
                                ts = ", ".join(p.get("tech_stack",[])[:4])
                                st.markdown(f"- **{p.get('name','')}** — {ts}")

            _cand_card(col_a, ca, "Candidate A")
            _cand_card(col_b, cb, "Candidate B")

            # ── LLM explanation ─────────────────────────────────────────────────
            if gen_exp:
                with st.spinner("Generating LLM comparison explanation…"):
                    try:
                        # use full ranked list context
                        all_sorted = sorted(
                            st.session_state.candidates,
                            key=lambda x: x["ats_score"], reverse=True
                        )
                        comp_result = generate_comparison_explanation(
                            jd_data=st.session_state.jd_data,
                            candidates=st.session_state.candidates,
                            ranked_results=all_sorted,
                        )
                        _add_lf(comp_result["langfuse"])
                        st.session_state.comparison = comp_result["explanation"]
                        _log("✅ LLM comparison generated")
                    except Exception as e:
                        st.error(f"LLM comparison failed: {e}")

            if st.session_state.comparison:
                st.markdown("---")
                st.markdown("### 🤖 LLM Recruiter Analysis")
                st.markdown(f"""
<div class="glass-card" style="border-left: 4px solid var(--accent);">
{st.session_state.comparison}
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – Observability (Langfuse)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_obs:
    st.markdown("## 🔭 Langfuse Observability")

    lf_enabled = lf_tracker.enabled
    if lf_enabled:
        st.success(f"✅ Langfuse connected → {os.getenv('LANGFUSE_HOST','')}")
    else:
        st.warning("⚠️  Langfuse not connected. Add keys in the sidebar to enable cloud tracking.")

    st.markdown("### 📊 Session Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total LLM Calls",   len(st.session_state.lf_logs))
    m2.metric("Total Tokens",      f"{st.session_state.total_tokens:,}")
    m3.metric("Est. Cost (USD)",   f"${st.session_state.total_cost:.4f}")
    m4.metric("Avg Tokens/Call",
              f"{int(st.session_state.total_tokens / max(len(st.session_state.lf_logs),1))}")

    if st.session_state.lf_logs:
        # ── Token usage chart ────────────────────────────────────────────────
        import plotly.graph_objects as go
        calls      = [f"Call {i+1}" for i in range(len(st.session_state.lf_logs))]
        in_toks    = [r.get("input_tokens", 0) for r in st.session_state.lf_logs]
        out_toks   = [r.get("output_tokens", 0) for r in st.session_state.lf_logs]

        tok_fig = go.Figure(data=[
            go.Bar(name="Input Tokens",  x=calls, y=in_toks,  marker_color="#6c63ff"),
            go.Bar(name="Output Tokens", x=calls, y=out_toks, marker_color="#00d2ff"),
        ])
        tok_fig.update_layout(
            barmode="stack",
            template="plotly_dark",
            paper_bgcolor="#0d0f16",
            plot_bgcolor="#151823",
            title="Token Usage per LLM Call",
            height=320,
            margin=dict(t=50,b=20,l=20,r=20),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(tok_fig, use_container_width=True)

        # ── Cost per call ─────────────────────────────────────────────────────
        costs = [r.get("cost_usd", 0) for r in st.session_state.lf_logs]
        cost_fig = go.Figure(go.Scatter(
            x=calls, y=costs,
            mode="lines+markers",
            line=dict(color="#ff6b9d", width=2),
            marker=dict(size=8, color="#ff6b9d"),
            fill="tozeroy",
            fillcolor="rgba(255,107,157,.15)",
        ))
        cost_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d0f16",
            plot_bgcolor="#151823",
            title="Estimated Cost per Call (USD)",
            height=280,
            margin=dict(t=50,b=20,l=20,r=20),
            yaxis=dict(tickformat=".5f", gridcolor="#2a2f45"),
            xaxis=dict(gridcolor="#2a2f45"),
        )
        st.plotly_chart(cost_fig, use_container_width=True)

        # ── Per-call details ──────────────────────────────────────────────────
        st.markdown("### 📋 LLM Call Log")
        for i, r in enumerate(st.session_state.lf_logs):
            with st.expander(
                f"Call {i+1} — {r.get('generation','')} | "
                f"tokens={r.get('total_tokens',0)} | "
                f"cost=${r.get('cost_usd',0):.5f}"
            ):
                st.markdown(f"""
<div class="lf-log-item">
  <b>Trace:</b> {r.get('trace_name','')} &nbsp;|&nbsp;
  <b>Generation:</b> {r.get('generation','')} &nbsp;|&nbsp;
  <b>Model:</b> {r.get('model','')}
</div>
<div class="lf-log-item">
  <b>Input tokens:</b> {r.get('input_tokens',0)} &nbsp;|&nbsp;
  <b>Output tokens:</b> {r.get('output_tokens',0)} &nbsp;|&nbsp;
  <b>Total:</b> {r.get('total_tokens',0)} &nbsp;|&nbsp;
  <b>Cost:</b> ${r.get('cost_usd',0):.5f}
</div>
<div class="lf-log-item">
  <b>Prompt chars:</b> {r.get('prompt_chars',0)} &nbsp;|&nbsp;
  <b>Response chars:</b> {r.get('response_chars',0)}
</div>
""", unsafe_allow_html=True)
                if r.get("metadata"):
                    st.json(r["metadata"])
    else:
        st.info("No LLM calls recorded yet. Process some resumes first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 – Processing Logs
# ═══════════════════════════════════════════════════════════════════════════════
with tab_logs:
    st.markdown("## 📜 Processing Logs")

    if st.session_state.processing_log:
        # Download button
        log_text = "\n".join(st.session_state.processing_log)
        st.download_button(
            "⬇️ Download Log",
            data=log_text,
            file_name="resume_parser_log.txt",
            mime="text/plain",
        )
        st.markdown("---")
        log_container = st.container()
        with log_container:
            for entry in reversed(st.session_state.processing_log):
                icon = "✅" if "✅" in entry else ("❌" if "❌" in entry else "")
                color = "#00e5a0" if "✅" in entry else ("#ff6b9d" if "❌" in entry else "#8890a6")
                st.markdown(
                    f'<div style="color:{color}; font-family:monospace; font-size:.85em; '
                    f'padding:3px 8px; border-left:3px solid {color}; margin:3px 0;">'
                    f'{entry}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("No logs yet. Start by analysing a job description and uploading resumes.")

    # ── File-based log viewer ─────────────────────────────────────────────────
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    if os.path.isdir(log_dir):
        log_files = sorted(os.listdir(log_dir), reverse=True)
        if log_files:
            st.markdown("---")
            st.markdown("### 📁 File Logs")
            sel_log = st.selectbox("Select log file", log_files)
            if sel_log:
                with open(os.path.join(log_dir, sel_log), "r") as f:
                    st.code(f.read(), language="text")

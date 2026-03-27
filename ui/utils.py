import time
import traceback
import streamlit as st
from typing import Dict, Any, List, Optional

# Constants
RANK_BADGES = ["1st", "2nd", "3rd"] + ["-"] * 20

def ui_log(msg: str, logger):
    """Log to session state and terminal."""
    ts = time.strftime("%H:%M:%S")
    if "processing_log" not in st.session_state:
        st.session_state.processing_log = []
    st.session_state.processing_log.append(f"[{ts}] {msg}")
    logger.info(msg)

def add_lf_record(record: dict):
    """Track Langfuse metrics in session state."""
    if record:
        if "lf_logs" not in st.session_state:
            st.session_state.lf_logs = []
        st.session_state.lf_logs.append(record)
        st.session_state.total_tokens += record.get("total_tokens", 0)
        st.session_state.total_cost += record.get("cost_usd", 0.0)

def get_score_color(score: float) -> str:
    """Map ATS score to color for visualization."""
    if score >= 80: return "#00e5a0"
    if score >= 60: return "#ffb340"
    return "#ff6b9d"

def safe_list(val) -> list:
    """Ensure value is return as a list."""
    if isinstance(val, list): return val
    if isinstance(val, str) and val: return [val]
    return []

def parse_uploaded_file(uploaded, extract_pdf_func, extract_docx_func, logger) -> Optional[Dict]:
    """Parse PDF or DOCX file using internal extractors."""
    raw = uploaded.read()
    name = uploaded.name.lower()
    try:
        if name.endswith(".pdf"):
            result = extract_pdf_func(raw)
        elif name.endswith(".docx"):
            result = extract_docx_func(raw)
        else:
            st.error(f"Unsupported format: {uploaded.name}")
            return None
        return result
    except Exception as e:
        st.error(f"Parse error ({uploaded.name}): {e}")
        logger.error(traceback.format_exc())
        return None

def build_candidate_dict(name: str, parse_result: dict, llm_data: dict, regex_data: dict,
                       gh: dict, lc: dict, jd_sim: float, skill_pct: float,
                       matched_skills: list, missing_skills: list, ats: dict) -> dict:
    """Merge extracted data into a standardized candidate dictionary."""
    llm_ft = llm_data.get("full_time_experience_years")
    llm_intern = llm_data.get("internship_months")
    llm_student = llm_data.get("is_student")
    llm_ctype = llm_data.get("candidate_type")

    ft_years = float(llm_ft if llm_ft is not None else regex_data.get("full_time_experience_years", 0))
    i_months = float(llm_intern if llm_intern is not None else regex_data.get("internship_months", 0))
    is_student = bool(llm_student if llm_student is not None else regex_data.get("is_student", False))
    ctype = llm_ctype or regex_data.get("candidate_type", "fresher")

    intern_w = round(i_months / 12.0 * 1.0, 2)
    effective_exp = round(ft_years + intern_w, 2)

    return {
        "name":               llm_data.get("name") or name,
        "email":              regex_data.get("email"),
        "phone":              regex_data.get("phone"),
        "github_url":         regex_data.get("github_url"),
        "linkedin_url":       regex_data.get("linkedin_url"),
        "leetcode_url":       regex_data.get("leetcode_url"),
        "summary":            llm_data.get("summary", ""),
        "skills":             safe_list(llm_data.get("skills")) or regex_data.get("tech_skills", []),
        "work_experience":    safe_list(llm_data.get("work_experience")),
        "projects":           safe_list(llm_data.get("projects")),
        "education":          safe_list(llm_data.get("education")),
        "certifications":     safe_list(llm_data.get("certifications")),
        "achievements":       safe_list(llm_data.get("achievements")),
        "full_time_exp_years": ft_years,
        "internship_months":   i_months,
        "effective_exp_years": effective_exp,
        "is_student":          is_student,
        "candidate_type":      ctype,
        "total_experience_years": effective_exp,
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

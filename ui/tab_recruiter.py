"""
ui/tab_recruiter.py  –  Recruiter Tools Tab

Exposes 3 recruiter-facing feature panels:
  Panel 1 – Timeline / Growth Analysis
  Panel 2 – Auto Email Generator
  Panel 3 – Skill Ontology Explorer
"""

import streamlit as st
from typing import Dict, Any, List, Optional


# ── helpers ───────────────────────────────────────────────────────────────────

def _pick_candidate(label: str = "Select a candidate") -> Optional[Dict[str, Any]]:
    """Shared candidate picker widget."""
    candidates = st.session_state.get("candidates", [])
    if not candidates:
        st.info("No candidates processed yet. Upload resumes in Tab 2 first.")
        return None
    names = [c["name"] for c in candidates]
    chosen = st.selectbox(label, names)
    return next(c for c in candidates if c["name"] == chosen)


def _score_bar(label: str, score: float, max_val: float = 100.0, color: str = "#4f8ef7"):
    """Compact labelled progress bar."""
    pct = min(score / max_val, 1.0)
    st.markdown(
        f"**{label}** &nbsp; `{score:.0f}/{max_val:.0f}`"
        f'<div style="background:#1e2130;border-radius:6px;height:10px;margin:4px 0 12px 0">'
        f'<div style="background:{color};width:{pct*100:.1f}%;height:10px;border-radius:6px"></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Panel 1 – Timeline ────────────────────────────────────────────────────────

_TIMELINE_ICONS = {
    "education":  "🎓",
    "project":    "🛠️",
    "internship": "🏢",
    "fulltime":   "💼",
}

_TIMELINE_COLORS = {
    "education":  "#60a5fa",
    "project":    "#34d399",
    "internship": "#fbbf24",
    "fulltime":   "#a78bfa",
}


def _render_timeline_panel():
    st.subheader("Timeline & Growth Analysis")
    cand = _pick_candidate("Analyze candidate timeline")
    if not cand:
        return

    use_llm = st.checkbox("Use LLM for narrative summary (1 LLM call)", value=True,
                          key="timeline_use_llm")

    if st.button("Generate Timeline", key="gen_timeline"):
        with st.spinner("Building timeline…"):
            from analyzers.timeline_analyzer import analyze_timeline
            result = analyze_timeline(cand, use_llm=use_llm)

        st.session_state["_timeline_result"] = result

    result = st.session_state.get("_timeline_result")
    if not result:
        return

    # Growth score
    st.divider()
    score = result["growth_score"]
    color = "#00e5a0" if score >= 70 else "#ffb340" if score >= 40 else "#ff6b9d"
    _score_bar("Growth Score", score, 100, color)

    st.caption(f"_Score breakdown: {result['score_breakdown']}_")

    # Narrative
    st.markdown(f"> **{result['growth_summary']}**")

    # Timeline events
    st.markdown("#### Career Timeline")
    timeline = result.get("timeline", [])
    if not timeline:
        st.warning("Not enough dated events found in this resume.")
        return

    for event in timeline:
        etype = event.get("type", "project")
        icon  = _TIMELINE_ICONS.get(etype, "📌")
        color = _TIMELINE_COLORS.get(etype, "#94a3b8")
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;margin-bottom:10px">'
            f'<div style="min-width:60px;font-weight:700;color:{color}">{event["year"]}</div>'
            f'<div style="border-left:2px solid {color};padding-left:12px;color:#e2e8f0">'
            f'{icon} {event["event"]}</div></div>',
            unsafe_allow_html=True,
        )


# ── Panel 2 – Email Generator ─────────────────────────────────────────────────

def _render_email_panel():
    st.subheader("✉️ Auto Email Generator")

    jd_data = st.session_state.get("jd_data") or {}
    role    = jd_data.get("role") or "the role"
    cand    = _pick_candidate("Generate email for candidate")
    if not cand:
        return

    col1, col2 = st.columns(2)
    with col1:
        email_role   = st.text_input("Role / Position", value=role, key="email_role")
        round_info   = st.text_input("Interview round (for invite)", value="Technical Round 1",
                                     key="email_round")
    with col2:
        use_llm = st.checkbox("Use LLM (richer emails, ~3 calls)", value=True, key="email_use_llm")
        email_types = st.multiselect(
            "Generate",
            ["Shortlist Email", "Rejection Email", "Interview Invite"],
            default=["Shortlist Email", "Rejection Email", "Interview Invite"],
            key="email_types",
        )

    if st.button("Generate Emails", key="gen_emails"):
        with st.spinner("Drafting emails…"):
            from extractors.email_generator import generate_all_emails
            ats_score = cand.get("ats_score", 0.0)
            emails    = generate_all_emails(
                candidate      = cand,
                role           = email_role,
                ats_score      = float(ats_score),
                matched_skills = cand.get("matched_skills", []),
                missing_skills = cand.get("missing_skills", []),
                round_info     = round_info,
                use_llm        = use_llm,
            )
        st.session_state["_emails_result"] = emails

    emails = st.session_state.get("_emails_result")
    if not emails:
        return

    st.divider()
    label_map = {
        "Shortlist Email":   "shortlist_email",
        "Rejection Email":   "rejection_email",
        "Interview Invite":  "interview_email",
    }
    tab_labels = [t for t in ["Shortlist Email", "Rejection Email", "Interview Invite"]
                  if t in email_types]

    if not tab_labels:
        return

    tabs = st.tabs(tab_labels)
    for tab, label in zip(tabs, tab_labels):
        with tab:
            key      = label_map[label]
            content  = emails.get(key, "")
            st.text_area(label, value=content, height=300, key=f"email_{key}")
            st.download_button(
                f"⬇ Download {label}",
                data=content,
                file_name=f"{cand['name'].replace(' ','_')}_{key}.txt",
                mime="text/plain",
                key=f"dl_{key}",
            )


# ── Panel 3 – Skill Ontology ──────────────────────────────────────────────────

_DOMAIN_COLORS = {
    "Frontend":        "#60a5fa",
    "Backend":         "#34d399",
    "AI/ML":           "#a78bfa",
    "DevOps":          "#fbbf24",
    "Cloud":           "#38bdf8",
    "Database":        "#f472b6",
    "Mobile":          "#fb923c",
    "Data_Engineering": "#4ade80",
    "Security":        "#f87171",
    "General_SE":      "#94a3b8",
}


def _render_ontology_panel():
    st.subheader("🧠 Skill Ontology Explorer")

    cand = _pick_candidate("Analyze skills for candidate")
    if not cand:
        return

    jd_data   = st.session_state.get("jd_data") or {}
    jd_skills = list(jd_data.get("required_skills") or [])
    role      = jd_data.get("role") or None

    if st.button("Analyze Skill Ontology", key="gen_ontology"):
        from analyzers.skill_ontology import analyze_skill_ontology
        with st.spinner("Mapping skills…"):
            result = analyze_skill_ontology(
                candidate          = cand,
                jd_required_skills = jd_skills or None,
                target_role        = role,
            )
        st.session_state["_ontology_result"] = result

    result = st.session_state.get("_ontology_result")
    if not result:
        return

    st.divider()
    domain_exp = result.get("domain_expertise", {})
    role_fit   = result.get("role_fit", {})
    gaps       = result.get("skill_gaps", [])
    norm_skills = result.get("normalized_skills", [])

    # Domain expertise grid
    st.markdown("#### Domain Expertise")
    cols = st.columns(3)
    for i, (dom, data) in enumerate(sorted(domain_exp.items(), key=lambda x: x[1]["score"], reverse=True)):
        score  = data["score"]
        skills = data["skills"]
        color  = _DOMAIN_COLORS.get(dom, "#94a3b8")
        with cols[i % 3]:
            st.markdown(
                f'<div style="background:#1e2130;border-radius:10px;padding:12px;margin-bottom:10px">'
                f'<div style="color:{color};font-weight:700;font-size:14px">{dom}</div>'
                f'<div style="font-size:22px;font-weight:800;color:#f8fafc">{"⭐"*score}{"☆"*(5-score)}&nbsp;{score}/5</div>'
                f'<div style="color:#94a3b8;font-size:12px;margin-top:4px">{", ".join(skills[:4])}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # Role fit
    st.markdown("#### Role Fit Scores")
    top_roles = list(role_fit.items())[:6]
    for role_name, pct in top_roles:
        color = "#00e5a0" if pct >= 75 else "#ffb340" if pct >= 50 else "#ff6b9d"
        _score_bar(role_name, pct, 100, color)

    # Skill gaps
    if gaps:
        st.markdown("#### ⚠️ Skill Gaps")
        gap_html = " ".join(
            f'<span style="background:#3f1f1f;color:#f87171;padding:3px 10px;'
            f'border-radius:20px;font-size:13px;margin:3px;display:inline-block">{g}</span>'
            for g in gaps
        )
        st.markdown(gap_html, unsafe_allow_html=True)

    # Normalized skills pill cloud
    with st.expander("View all normalized skills"):
        pills = " ".join(
            f'<span style="background:#1e3a5f;color:#93c5fd;padding:2px 8px;'
            f'border-radius:14px;font-size:12px;margin:2px;display:inline-block">{s}</span>'
            for s in norm_skills
        )
        st.markdown(pills, unsafe_allow_html=True)

    if result.get("unmapped_skills"):
        st.caption(f"Unmapped skills (not in ontology): {', '.join(result['unmapped_skills'][:10])}")


# ── Tab entry point ───────────────────────────────────────────────────────────

def render_tab_recruiter():
    """Main entry — called from app.py."""
    st.header("Recruiter Tools")

    if not st.session_state.get("candidates"):
        st.warning("Process resumes in the **Upload Resumes** tab first.")
        return

    panel = st.radio(
        "Select Tool",
        ["Timeline & Growth", "Auto Email Generator", "Skill Ontology"],
        horizontal=True,
        key="recruiter_panel",
        label_visibility="collapsed",
    )

    st.divider()

    if panel == "Timeline & Growth":
        _render_timeline_panel()
    elif panel == "Auto Email Generator":
        _render_email_panel()
    elif panel == "Skill Ontology":
        _render_ontology_panel()

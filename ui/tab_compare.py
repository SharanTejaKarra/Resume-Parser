import streamlit as st
from extractors.llm_extractor import generate_comparison_explanation
from ui.utils import get_score_color, add_lf_record, ui_log

def render_tab_compare(log):
    """Render the Candidate Comparison tab — strictly 2 candidates side-by-side."""
    st.header("Candidate Comparison")

    candidates = st.session_state.get("candidates", [])

    if len(candidates) < 2:
        st.info("Upload at least 2 resumes to compare.")
        return

    names = [c["name"] for c in candidates]

    with st.form("compare_form"):
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            c1_name = st.selectbox("Candidate A", names, index=0, key="compare_sel_a")
        with col_sel2:
            # Default to a different candidate than A
            default_b = 1 if len(names) > 1 else 0
            c2_name = st.selectbox("Candidate B", names, index=default_b, key="compare_sel_b")

        gen_exp   = st.checkbox("Generate LLM explanation (compares only these 2)", value=True)
        submitted = st.form_submit_button("Compare", use_container_width=True)

    if submitted:
        if c1_name == c2_name:
            st.warning("Please select two **different** candidates to compare.")
            return

        ca = next(c for c in candidates if c["name"] == c1_name)
        cb = next(c for c in candidates if c["name"] == c2_name)

        st.divider()
        col_a, col_b = st.columns(2)

        _render_compare_card(col_a, ca, "Candidate A")
        _render_compare_card(col_b, cb, "Candidate B")

        if gen_exp:
            with st.spinner("Generating LLM comparison explanation…"):
                try:
                    # Pass ONLY the 2 selected candidates — not the full list
                    two_candidates = sorted([ca, cb], key=lambda x: x["ats_score"], reverse=True)
                    comp_result = generate_comparison_explanation(
                        jd_data=st.session_state.jd_data,
                        candidates=two_candidates,
                        ranked_results=two_candidates,
                    )
                    add_lf_record(comp_result["langfuse"])
                    st.session_state.comparison = comp_result["explanation"]
                    ui_log("LLM comparison generated", log)
                except Exception as e:
                    st.error(f"LLM comparison failed: {e}")

        if st.session_state.comparison:
            st.divider()
            st.subheader("LLM Recruiter Analysis")
            st.info(st.session_state.comparison)


def _render_compare_card(col, cand, label):
    """Internal helper to render a comparison card for a candidate."""
    score = cand["ats_score"]
    color = "#00e5a0" if score >= 70 else "#ffb340" if score >= 50 else "#ff6b9d"

    with col:
        # Header card
        with st.container(border=True):
            st.subheader(label)
            st.caption(cand["name"])
            st.markdown(
                f'<div style="font-size:36px;font-weight:800;color:{color}">'
                f'{score}<span style="font-size:16px;color:#94a3b8"> / 100</span></div>',
                unsafe_allow_html=True,
            )

        # Core metrics
        m1, m2 = st.columns(2)
        m1.metric("JD Similarity",  f"{cand['jd_similarity']:.3f}")
        m2.metric("Skill Match",    f"{cand['skill_match_pct']:.1f}%")

        m3, m4 = st.columns(2)
        m3.metric("Experience",     f"{cand['total_experience_years']} yrs")
        m4.metric("GitHub Score",   cand["github"].get("github_score", 0))

        m5, m6 = st.columns(2)
        m5.metric("LeetCode Score", cand["leetcode"].get("leetcode_score", 0))
        m6.metric("Scoring Mode",   cand.get("ats_breakdown", {}).get("scoring_mode", "—").capitalize())

        # Skill pills
        st.markdown("**Matched Skills**")
        matched = cand.get("matched_skills", [])[:10]
        if matched:
            pills = " ".join(
                f'<span style="background:#1e3a2f;color:#4ade80;padding:2px 8px;'
                f'border-radius:12px;font-size:12px;margin:2px;display:inline-block">{s}</span>'
                for s in matched
            )
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("None")

        st.markdown("**Missing JD Skills**")
        missing = cand.get("missing_skills", [])[:8]
        if missing:
            pills = " ".join(
                f'<span style="background:#3f1a1a;color:#f87171;padding:2px 8px;'
                f'border-radius:12px;font-size:12px;margin:2px;display:inline-block">{s}</span>'
                for s in missing
            )
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("None")

        # Experience entries
        if cand.get("work_experience"):
            st.markdown("**Experience**")
            for ex in cand["work_experience"][:3]:
                if isinstance(ex, dict):
                    role    = ex.get("title") or ex.get("role", "")
                    company = ex.get("company", "")
                    start   = ex.get("start", "")
                    end     = ex.get("end", "")
                    intern_tag = " *(Intern)*" if ex.get("is_internship") else ""
                    st.markdown(f"- **{role}**{intern_tag} @ {company}  \n  `{start} – {end}`")

        # Top projects
        if cand.get("projects"):
            st.markdown("**Projects**")
            for p in cand["projects"][:3]:
                if isinstance(p, dict):
                    ts = ", ".join((p.get("tech_stack") or [])[:4])
                    st.markdown(f"- **{p.get('name','')}**  \n  `{ts}`")

import streamlit as st
from extractors.llm_extractor import generate_comparison_explanation
from ui.utils import get_score_color, add_lf_record, ui_log

def render_tab_compare(log):
    """Render the Candidate Comparison tab."""
    st.header("Candidate Comparison")

    if len(st.session_state.candidates) < 2:
        st.info("Upload at least 2 resumes to compare.")
        return

    names = [c["name"] for c in st.session_state.candidates]

    with st.form("compare_form"):
        c1_name = st.selectbox("Candidate A", names, index=0)
        c2_name = st.selectbox("Candidate B", names, index=min(1, len(names)-1))
        gen_exp  = st.checkbox("Generate LLM explanation", value=True)
        submitted = st.form_submit_button("Compare")

    if submitted:
        ca = next(c for c in st.session_state.candidates if c["name"] == c1_name)
        cb = next(c for c in st.session_state.candidates if c["name"] == c2_name)

        st.divider()
        col_a, col_b = st.columns(2)

        _render_compare_card(col_a, ca, "Candidate A")
        _render_compare_card(col_b, cb, "Candidate B")

        if gen_exp:
            with st.spinner("Generating LLM comparison explanation…"):
                try:
                    all_sorted = sorted(st.session_state.candidates, key=lambda x: x["ats_score"], reverse=True)
                    comp_result = generate_comparison_explanation(
                        jd_data=st.session_state.jd_data,
                        candidates=st.session_state.candidates,
                        ranked_results=all_sorted,
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
    with col:
        with st.container(border=True):
            st.subheader(label)
            st.caption(cand['name'])
            st.metric("ATS Score", cand['ats_score'])

        st.metric("JD Similarity",  f"{cand['jd_similarity']:.3f}")
        st.metric("Skill Match",    f"{cand['skill_match_pct']:.1f}%")
        st.metric("Experience",     f"{cand['total_experience_years']} yrs")
        st.metric("GitHub Score",   cand["github"].get("github_score", 0))
        st.metric("LeetCode Score", cand["leetcode"].get("leetcode_score", 0))

        st.write("**Matched Skills:**")
        st.write(", ".join(cand["matched_skills"][:12]))

        st.write("**Missing JD Skills:**")
        st.write(", ".join([f"Missing: {s}" for s in cand["missing_skills"][:8]]))

        if cand["work_experience"]:
            st.write("**Experience:**")
            for ex in cand["work_experience"][:3]:
                if isinstance(ex, dict):
                    st.write(f"- **{ex.get('title','')}** @ {ex.get('company','')} ({ex.get('start','')}–{ex.get('end','')})")

        if cand["projects"]:
            st.write("**Projects:**")
            for p in cand["projects"][:3]:
                if isinstance(p, dict):
                    ts = ", ".join(p.get("tech_stack",[])[:4])
                    st.write(f"- **{p.get('name','')}** — {ts}")

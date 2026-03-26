import streamlit as st
import traceback
from extractors.llm_extractor import extract_jd_llm
from ui.utils import ui_log, add_lf_record, safe_list

def render_tab_jd(log):
    """Render the Job Description analysis tab."""
    st.header("Step 1 — Paste Job Description")
    st.write("The system will extract required skills, experience, and role details.")

    jd_input = st.text_area(
        "Job Description",
        height=300,
        placeholder="Paste the full job description here…",
        value=st.session_state.jd_text,
        key="jd_textarea",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        parse_jd = st.button("Analyse JD", use_container_width=True)

    if parse_jd and jd_input.strip():
        st.session_state.jd_text = jd_input
        with st.spinner("Parsing job description with LLM…"):
            try:
                jd_result = extract_jd_llm(jd_input)
                st.session_state.jd_data = jd_result["data"]
                add_lf_record(jd_result["langfuse"])
                ui_log("JD parsed successfully", log)
                st.success("Job description analysed!")
            except Exception as e:
                st.error(f"JD parsing failed: {e}")
                log.error(traceback.format_exc())

    if st.session_state.jd_data:
        jd = st.session_state.jd_data
        st.divider()
        st.subheader("Extracted JD Fields")

        c1, c2, c3 = st.columns(3)
        c1.metric("Role",         jd.get("role", "—"))
        c2.metric("Min. Experience", f"{jd.get('min_experience_years', 0)} yrs")
        c3.metric("Employment",   jd.get("employment_type", "—"))

        col_req, col_pref = st.columns(2)
        with col_req:
            st.write("**Required Skills**")
            for s in safe_list(jd.get("required_skills")):
                st.write(f"Required: {s}")

        with col_pref:
            st.write("**Preferred Skills**")
            for s in safe_list(jd.get("preferred_skills")):
                st.write(f"Preferred: {s}")

        with st.expander("Full JD JSON"):
            st.json(jd)

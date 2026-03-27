import streamlit as st
import traceback
from extractors.regex_extractor import extract_regex_fields
from extractors.llm_extractor import extract_resume_llm
from analyzers.github_analyzer import analyze_github
from analyzers.leetcode_analyzer import analyze_leetcode
from scoring.embedding_matcher import compute_jd_similarity, compute_skill_match
from scoring.ats_scorer import compute_ats_score
from ui.utils import ui_log, add_lf_record, safe_list, parse_uploaded_file, build_candidate_dict

def safe_float(val, default=0.0):
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default

def render_tab_upload(log, extract_pdf, extract_docx):
    """Render the Resume Upload & Processing tab."""
    st.header("Step 2 — Upload Resumes")

    if not st.session_state.jd_data:
        st.warning("Please analyse a Job Description first (Tab 1).")
        return

    uploaded_files = st.file_uploader(
        "Drop PDF or DOCX resumes here (multiple supported)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="resume_uploader",
    )

    enable_gh = st.session_state.get("sidebar_gh", True)
    enable_lc = st.session_state.get("sidebar_lc", True)

    if uploaded_files:
        if st.button("Process All Resumes", width="stretch"):
            jd     = st.session_state.jd_data
            jd_txt = st.session_state.jd_text

            new_candidates = []
            prog = st.progress(0)
            status_box = st.empty()

            for idx, uf in enumerate(uploaded_files):
                prog.progress((idx) / len(uploaded_files))
                status_box.info(f"Processing **{uf.name}** ({idx+1}/{len(uploaded_files)})…")
                ui_log(f"Starting: {uf.name}", log)

                try:
                    # 1. Parse document
                    parse_res = parse_uploaded_file(uf, extract_pdf, extract_docx, log)
                    if not parse_res: continue

                    parse_status = parse_res.get("parse_status", "OK")
                    parser_used  = parse_res.get("parser_used", "unknown")
                    char_count   = parse_res.get("char_count", 0)
                    full_text    = parse_res["full_text"]
                    col_type     = "Two-column" if parse_res["is_two_column"] else "Single-column"

                    ui_log(f"  Parsed ({col_type}, {parse_res['pages']} pages, {char_count} chars) via {parser_used} [{parse_status}]", log)

                    if parse_status == "PARSE_FAILED":
                        st.warning(f"**{uf.name}** — all PDF parsers returned insufficient text. Skipping.")
                        ui_log(f"  PARSE_FAILED – skipping {uf.name}", log)
                        continue

                    if parse_status == "LOW_CONFIDENCE":
                        st.warning(f"**{uf.name}** — low text confidence. Results may be inaccurate.")

                    # 2. Regex
                    regex = extract_regex_fields(full_text)

                    # 3. LLM
                    cand_label = uf.name.replace(".pdf","").replace(".docx","")
                    llm_res    = extract_resume_llm(full_text, candidate_name=cand_label)
                    llm_data   = llm_res["data"]
                    add_lf_record(llm_res["langfuse"])

                    # 4. GitHub
                    gh = {"github_score": 0.0, "error": "disabled"}
                    if enable_gh and regex.get("github_username"):
                        gh = analyze_github(regex["github_username"])

                    # 5. LeetCode
                    lc = {"leetcode_score": 0.0, "error": "disabled"}
                    if enable_lc and regex.get("leetcode_username"):
                        lc = analyze_leetcode(regex["leetcode_username"])

                    # 6. Embedding Similarity
                    jd_sim = compute_jd_similarity(full_text, jd_txt)

                    # 7. Skill Match
                    all_skills = sorted(set(safe_list(llm_data.get("skills")) + regex.get("tech_skills", [])))
                    jd_req_skills = safe_list(jd.get("required_skills"))
                    skill_pct, matched, missing = compute_skill_match(all_skills, jd_req_skills)

                    # 8. ATS Score
                    ui_log(f"FT Exp LLM: {llm_data.get('full_time_experience_years')}", log)
                    ui_log(f"FT Exp Regex: {regex.get('full_time_experience_years')}", log)

                    ft_exp = max(
                        safe_float(llm_data.get("full_time_experience_years")),
                        safe_float(regex.get("full_time_experience_years"))
                    )

                    i_months = max(
                        safe_float(llm_data.get("internship_months")),
                        safe_float(regex.get("internship_months"))
                    )
                    
                    is_stud = bool(llm_data.get("is_student") if llm_data.get("is_student") is not None else regex.get("is_student", False))
                    ctype = llm_data.get("candidate_type") or regex.get("candidate_type", "fresher")

                
                    ats = compute_ats_score(
                        jd_similarity=jd_sim,
                        skill_match_pct=skill_pct,
                        full_time_exp_years=ft_exp,
                        internship_months=i_months,
                        is_student=is_stud,
                        candidate_type=ctype,
                        min_exp_required=float(jd.get("min_experience_years") or 0),
                        github_score=float(gh.get("github_score", 0)),
                        leetcode_score=float(lc.get("leetcode_score", 0)),
                        projects=safe_list(llm_data.get("projects")),
                    )




                    # 9. Build
                    candidate = build_candidate_dict(
                        name=cand_label, parse_result=parse_res,
                        llm_data=llm_data, regex_data=regex,
                        gh=gh, lc=lc, jd_sim=jd_sim, skill_pct=skill_pct,
                        matched_skills=matched, missing_skills=missing, ats=ats
                    )
                    new_candidates.append(candidate)
                    ui_log(f"  ATS score: {ats['ats_score']} (mode={ats['scoring_mode']})", log)

                except Exception as e:
                    st.error(f"Error processing {uf.name}: {e}")
                    log.error(traceback.format_exc())
                    ui_log(f"  Error: {e}", log)

            prog.progress(1.0)
            status_box.empty()

            existing_names = {c["name"] for c in st.session_state.candidates}
            for c in new_candidates:
                if c["name"] not in existing_names:
                    st.session_state.candidates.append(c)

            st.session_state.ranked = sorted(st.session_state.candidates, key=lambda c: c["ats_score"], reverse=True)
            st.session_state.comparison = None
            st.success(f"Processed {len(new_candidates)} resume(s)!")
            st.balloons()

    # Show processed candidates
    if st.session_state.candidates:
        st.divider()
        st.subheader(f"Processed Candidates ({len(st.session_state.candidates)})")
        # Sort candidates by ATS score descending
        sorted_candidates = sorted(st.session_state.candidates, key=lambda x: x.get("ats_score", 0), reverse=True)
        for idx, c in enumerate(sorted_candidates):
            render_candidate_expander(c, idx)

def render_candidate_expander(c, idx):
    """Render the detailed expander for a candidate."""
    parse_badge = {"OK": "OK", "LOW_CONFIDENCE": "Low Confidence", "PARSE_FAILED": "Failed"}.get(c.get("parse_status", "OK"), "OK")
    ctype_emoji = {"student": "Student", "fresher": "Fresher", "experienced": "Experienced"}.get(c.get("candidate_type", "fresher"), "Experienced")

    with st.expander(f"**{c['name']}**  |  ATS: {c['ats_score']}  |  {ctype_emoji}  |  Parse: {parse_badge}"):
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("ATS Score",     f"{c['ats_score']}")
        col_b.metric("JD Similarity", f"{c['jd_similarity']:.2f}")
        col_c.metric("Skill Match",   f"{c['skill_match_pct']:.1f}%")
        col_d.metric("Scoring Mode",  c.get('ats_breakdown', {}).get('scoring_mode', '—').capitalize())

        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Full-time exp",   f"{c.get('full_time_exp_years', 0):.2f} yrs")
        ec2.metric("Internship",       f"{c.get('internship_months', 0):.0f} months")
        ec3.metric("Effective exp",    f"{c.get('effective_exp_years', 0):.2f} yrs")

        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.markdown(f"**Parser:** `{c.get('parser_used','—')}`")
        pcol2.markdown(f"**Layout:** {'Two-column' if c['is_two_column'] else 'Single-column'}")
        pcol3.markdown(f"**Text chars:** `{c.get('char_count', 0):,}`")

        st.write("**Matched Skills:**")
        st.write(", ".join(c["matched_skills"]))

        st.write("**Missing Skills:**")
        st.write(", ".join(c["missing_skills"]))



        if c["github"].get("public_repos"):
            gh_d = c["github"]
            st.write(f"**GitHub:** [{gh_d['username']}]({gh_d['profile_url']}) · {gh_d['public_repos']} repos · Stars: {gh_d['total_stars']} · Score: **{gh_d['github_score']}**")
        if c["leetcode"].get("total_solved"):
            lc_d = c["leetcode"]
            st.write(f"**LeetCode:** [{lc_d['username']}]({lc_d['profile_url']}) · Easy: {lc_d['easy_solved']} Medium: {lc_d['medium_solved']} Hard: {lc_d['hard_solved']} · Score: **{lc_d['leetcode_score']}**")

        if st.checkbox("Show Full Extracted Data", key=f"raw_data_{idx}"):
            clean = {k: v for k, v in c.items() if k not in ["github", "leetcode", "ats_breakdown"]}
            st.json(clean)

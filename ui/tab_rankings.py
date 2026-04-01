import streamlit as st
from ui.utils import get_score_color, get_ordinal
from ui.components import render_ats_bar_chart, render_skills_radar, render_candidate_row

def render_tab_rankings():
    """Render the Candidate Rankings tab."""
    st.header("Candidate Rankings")

    if not st.session_state.ranked:
        st.info("Process at least one resume to see rankings.")
    else:
        ranked = st.session_state.ranked
        
        # 1. Bar Chart
        render_ats_bar_chart(ranked)

        # v2: Confidence Summary
        if ranked and ranked[0].get("confidence"):
            st.subheader("Confidence Overview")
            conf_cols = st.columns(min(len(ranked), 5))
            for i, c in enumerate(ranked[:5]):
                conf = c.get("confidence", {})
                rec = conf.get("recommendation", "unknown")
                rec_colors = {"strong_candidate": "#00e5a0", "review_recommended": "#ffb340",
                             "proceed_with_caution": "#ff6b9d", "flag_for_review": "#ef4444"}
                color = rec_colors.get(rec, "#94a3b8")
                with conf_cols[i]:
                    st.markdown(
                        f'<div style="text-align:center;padding:8px;background:#f8f7f2;border-radius:8px;border-top:3px solid {color}">'
                        f'<div style="font-size:11px;color:#6d6a5a">{c["name"][:15]}</div>'
                        f'<div style="font-size:20px;font-weight:700;color:{color}">{conf.get("confidence_score", 0):.0f}</div>'
                        f'<div style="font-size:10px;color:#94a3b8">{c.get("candidate_level", "—").upper()}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # 2. Radar Chart
        render_skills_radar(ranked)

        # 3. Detailed Ranking Table
        st.subheader("Detailed Ranking Table")
        for i, c in enumerate(ranked):
            badge = get_ordinal(i + 1)
            color = get_score_color(c["ats_score"])
            render_candidate_row(c, badge, color)

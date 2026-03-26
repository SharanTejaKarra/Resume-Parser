import streamlit as st
from ui.utils import get_score_color, RANK_BADGES
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

        # 2. Radar Chart
        render_skills_radar(ranked)

        # 3. Detailed Ranking Table
        st.subheader("📋 Detailed Ranking Table")
        for i, c in enumerate(ranked):
            badge = RANK_BADGES[i] if i < len(RANK_BADGES) else "🎖️"
            color = get_score_color(c["ats_score"])
            render_candidate_row(c, badge, color)

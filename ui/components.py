import streamlit as st
import plotly.graph_objects as go
from ui.utils import get_score_color

def render_hero_banner():
    """Render the main hero banner."""
    st.title("Resume Engine AI")
    st.write("Column-aware PDF parsing, LLM-based semantic extraction, embedding job matching, GitHub and LeetCode enrichment, and ATS scoring with built-in Langfuse observability.")

def render_ats_bar_chart(ranked):
    """Render the ATS score bar chart."""
    names  = [c["name"] for c in ranked]
    scores = [c["ats_score"] for c in ranked]
    
    fig = go.Figure(go.Bar(
        x=names, y=scores,
        marker=dict(
            color=scores,
            colorscale=[[0, "#ff6b9d"], [0.5, "#ffb340"], [1, "#00e5a0"]],
            cmin=0, cmax=100,
            showscale=True,
            colorbar=dict(title="ATS"),
        ),
        text=[f"{s}" for s in scores],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>ATS: %{y}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="ATS Score Ranking", font=dict(size=18, color="#3d3a2a")),
        yaxis=dict(range=[0, 110], gridcolor="#d6d2c4", tickfont=dict(color="#6d6a5a")),
        xaxis=dict(gridcolor="#d6d2c4", tickfont=dict(color="#6d6a5a")),
        margin=dict(t=60, b=20, l=20, r=20),
        height=380,
    )
    st.plotly_chart(fig, width="stretch")

def render_skills_radar(ranked):
    """Render the skills radar chart for top 3 candidates."""
    if len(ranked) < 2:
        return
        
    st.subheader("Skills Radar - Top 3")
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
            bgcolor="#ffffff",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#d6d2c4", tickfont=dict(color="#6d6a5a")),
            angularaxis=dict(gridcolor="#d6d2c4", tickfont=dict(color="#6d6a5a")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        template="plotly_white",
        height=420,
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(radar_fig, width="stretch")

def render_candidate_row(c, badge, color):
    """Render a single candidate ranking card."""
    with st.container(border=True):
        col_rank, col_score, col_info = st.columns([1, 1, 4])
        with col_rank:
            st.subheader(badge)
        with col_score:
            st.markdown(f"### <span style='color:{color}'>{c['ats_score']}</span>", unsafe_allow_html=True)
        with col_info:
            st.write(f"**{c['name']}**")
            st.caption(f"{c['email'] or '—'} · {c['total_experience_years']} yrs exp")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("JD Sim", f"{c['jd_similarity']:.3f}")
        c2.metric("Skill", f"{c['skill_match_pct']:.1f}%")
        c3.metric("GitHub", f"{c['github'].get('github_score',0):.1f}")
        c4.metric("LeetCode", f"{c['leetcode'].get('leetcode_score',0):.1f}")
        c5.metric("Projects", f"{c['ats_breakdown']['project_score']:.1f}")

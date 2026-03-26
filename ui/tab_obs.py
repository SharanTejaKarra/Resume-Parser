import os
import streamlit as st
import plotly.graph_objects as go
from utils.langfuse_tracker import tracker as lf_tracker

def render_tab_observability():
    """Render the Langfuse Observability tab."""
    st.header("Langfuse Observability")

    lf_enabled = lf_tracker.enabled
    if lf_enabled:
        st.success(f"Langfuse connected -> {os.getenv('LANGFUSE_HOST','')}")
    else:
        st.warning("Langfuse not connected. Add keys in the sidebar to enable cloud tracking.")

    st.subheader("Session Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total LLM Calls",   len(st.session_state.lf_logs))
    m2.metric("Total Tokens",      f"{st.session_state.total_tokens:,}")
    m3.metric("Est. Cost (USD)",   f"${st.session_state.total_cost:.4f}")
    m4.metric("Avg Tokens/Call", f"{int(st.session_state.total_tokens / max(len(st.session_state.lf_logs),1))}")

    if st.session_state.lf_logs:
        # 1. Token chart
        calls = [f"Call {i+1}" for i in range(len(st.session_state.lf_logs))]
        in_toks = [r.get("input_tokens", 0) for r in st.session_state.lf_logs]
        out_toks = [r.get("output_tokens", 0) for r in st.session_state.lf_logs]

        tok_fig = go.Figure(data=[
            go.Bar(name="Input Tokens",  x=calls, y=in_toks,  marker_color="#bb5a38"),
            go.Bar(name="Output Tokens", x=calls, y=out_toks, marker_color="#e2a48d"),
        ])
        tok_fig.update_layout(
            barmode="stack", template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text="Token Usage per LLM Call", font=dict(color="#3d3a2a")),
            height=320, margin=dict(t=50,b=20,l=20,r=20),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#3d3a2a")),
            xaxis=dict(tickfont=dict(color="#6d6a5a")), yaxis=dict(tickfont=dict(color="#6d6a5a"), gridcolor="#d6d2c4"),
        )
        st.plotly_chart(tok_fig, use_container_width=True)

        # 2. Cost chart
        costs = [r.get("cost_usd", 0) for r in st.session_state.lf_logs]
        cost_fig = go.Figure(go.Scatter(
            x=calls, y=costs, mode="lines+markers", line=dict(color="#d97d5c", width=2),
            marker=dict(size=8, color="#d97d5c"), fill="tozeroy", fillcolor="rgba(217,125,92,.15)",
        ))
        cost_fig.update_layout(
            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text="Estimated Cost per Call (USD)", font=dict(color="#3d3a2a")),
            height=280, margin=dict(t=50,b=20,l=20,r=20),
            yaxis=dict(tickformat=".5f", gridcolor="#d6d2c4", tickfont=dict(color="#6d6a5a")),
            xaxis=dict(gridcolor="#d6d2c4", tickfont=dict(color="#6d6a5a")),
        )
        st.plotly_chart(cost_fig, use_container_width=True)

        st.subheader("LLM Call Log")
        for i, r in enumerate(st.session_state.lf_logs):
            with st.expander(f"Call {i+1} — {r.get('generation','')} | tokens={r.get('total_tokens',0)} | cost=${r.get('cost_usd',0):.5f}"):
                st.info(f"**Trace:** {r.get('trace_name','')} | **Model:** {r.get('model','')}")
                st.code(f"Input tokens: {r.get('input_tokens',0)} | Output tokens: {r.get('output_tokens',0)} | Total: {r.get('total_tokens',0)}")
                st.code(f"Prompt chars: {r.get('prompt_chars',0)} | Response chars: {r.get('response_chars',0)}")
                if r.get("metadata"): st.json(r["metadata"])
    else:
        st.info("No LLM calls recorded yet.")

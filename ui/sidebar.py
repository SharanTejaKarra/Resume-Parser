import os
import streamlit as st

def render_sidebar():
    """Render the Streamlit sidebar with stats and configuration."""
    with st.sidebar:
        # Active config (read-only)
        st.subheader("Active Config")
        _ol_model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
        _has_lf   = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))

        st.write(f"""
| Setting | Value |
|---|---|
| **Host** | `Ollama` |
| **Model** | `{_ol_model}` |
| **Langfuse** | {'connected' if _has_lf else 'not set'} |
        """)

        st.divider()

        # Enrichment toggles
        st.subheader("Enrichment")
        _enrich_gh = st.toggle("GitHub",   value=True, key="sidebar_gh")
        _enrich_lc = st.toggle("LeetCode", value=True, key="sidebar_lc")

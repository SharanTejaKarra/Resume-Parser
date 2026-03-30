import os
import streamlit as st

def render_sidebar():
    """Render the Streamlit sidebar with stats and configuration."""
    with st.sidebar:
        # Active config (read-only)
        st.subheader("Active Config")

        _provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
        _has_lf   = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))

        if _provider == "openai":
            _model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            _provider_label = "OpenAI"
        else:
            _model_name = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
            _provider_label = "Ollama"

        st.write(f"""
| Setting | Value |
|---|---|
| **Provider** | `{_provider_label}` |
| **Model** | `{_model_name}` |
| **Langfuse** | {'Connected' if _has_lf else 'Not Set'} |
        """)

        st.divider()

        # Enrichment toggles
        st.subheader("Enrichment")
        _enrich_gh = st.toggle("GitHub",   value=True, key="sidebar_gh")
        _enrich_lc = st.toggle("LeetCode", value=True, key="sidebar_lc")

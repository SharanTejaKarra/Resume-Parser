import streamlit as st

def render_tab_logs():
    """Render the human-readable processing logs tab."""
    st.header("Processing Logs")
    if not st.session_state.processing_log:
        st.info("Logs will appear here once processing starts.")
    else:
        for entry in reversed(st.session_state.processing_log):
            st.code(entry)

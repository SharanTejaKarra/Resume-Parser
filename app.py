"""
app.py  -  AI Resume Parser  |  Modular Orchestrator
"""
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from utils.logger import get_logger
from parsers.pdf_parser import extract_pdf, extract_pdf_pdfplumber_fallback
from parsers.docx_parser import extract_docx

# UI Components & Tabs
from ui.sidebar import render_sidebar
from ui.components import render_hero_banner
from ui.tab_jd import render_tab_jd
from ui.tab_upload import render_tab_upload
from ui.tab_rankings import render_tab_rankings
from ui.tab_compare import render_tab_compare
from ui.tab_obs import render_tab_observability
from ui.tab_logs import render_tab_logs
from ui.tab_recruiter import render_tab_recruiter

log = get_logger("app")

# Page config
st.set_page_config(
    page_title="Resume Engine AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
def init_state():
    defaults = {
        "jd_data":          None,
        "jd_text":          "",
        "candidates":       [],
        "ranked":           [],
        "comparison":       None,
        "lf_logs":          [],
        "processing_log":   [],
        "total_tokens":     0,
        "total_cost":       0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Sidebar
render_sidebar()

# Main UI
render_hero_banner()

tab_jd, tab_upload, tab_rank, tab_compare, tab_recruiter = st.tabs([
    "Job Description",
    "Upload Resumes",
    "Rankings",
    "Compare",
    "Recruiter Tools",
    # "Observability",
    # "Logs",
])

with tab_jd:
    render_tab_jd(log)

with tab_upload:
    render_tab_upload(log, extract_pdf, extract_docx)

with tab_rank:
    render_tab_rankings()

with tab_compare:
    render_tab_compare(log)

with tab_recruiter:
    render_tab_recruiter()

# with tab_obs:
#     render_tab_observability()

# with tab_logs:
#     render_tab_logs()

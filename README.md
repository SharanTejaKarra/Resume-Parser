# Resume Engine AI

A professional AI-powered resume parsing and ATS scoring engine. It extracts structured data from complex layouts and provides deep candidate insights.

## Features
- **Multi-Stage Parsing**: Cascading extraction for complex PDF layouts.
- **Student-Aware Scoring**: Intelligent ATS weighting for both students and professionals.
- **Profile Enrichment**: Integrated GitHub and LeetCode analytics.
- **Observability**: Built-in Langfuse tracking for tokens and costs.
- **Modern UI**: Clean, native Streamlit interface.

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repo-url>
cd Resume_Engine_AI

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file from the template:
```bash
cp .env.example .env
```
Add your `OLLAMA_MODEL`, `GITHUB_TOKEN`, and `LANGFUSE` keys.

### 3. Run
```bash
streamlit run app.py
```

## Project Structure
- `ui/`: Modular Streamlit interface components.
- `parsers/`: PDF and DOCX extraction logic.
- `extractors/`: LLM and Regex data retrieval.
- `analyzers/`: GitHub and LeetCode enrichment.
- `scoring/`: ATS ranking and matching algorithms.
- `utils/`: Logging and observability trackers.

## License
MIT

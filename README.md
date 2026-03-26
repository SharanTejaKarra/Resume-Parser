# 🧠 AI Resume Parser & ATS Scorer

A production-grade, highly robust AI resume parser and ATS scoring system built with Python, Streamlit, and Local/Cloud LLMs (via Ollama). This system is designed to accurately parse complex resume layouts (including multi-column PDFs), semantically extract skills and experiences, mitigate common matching biases (especially for students vs. experienced professionals), and provide a rich scoring dashboard with GitHub and LeetCode enrichments.

## ✨ Key Features

*   **Robust Multi-Stage PDF Parsing**: Automatically cascades through `PyMuPDF` → `pdfplumber` → `pdfminer.six` to ensure text is extracted accurately, recognizing complex 2-column layouts without jumbling text.
*   **Student-Aware ATS Scoring**: Smartly identifies students vs. experienced candidates, avoiding penalizing undergraduates natively. Internships and Full-time roles are split, weighted, and evaluated accurately.
*   **Semantic Skill Inference**: Maps generic terminologies (e.g. `LangGraph`, `LlamaIndex`) to categorized structural domain abilities (e.g., `LLM Orchestration`, `NLP`).
*   **Comprehensive Data Extraction**: Uses deterministic regex extractions for safe values (URLs, emails, phone numbers) combined with intelligent LLM JSON parsing for experiences, projects, and summaries.
*   **Developer Profile Enrichment**: Integrates directly with GitHub and LeetCode APIs to fetch and score candidate coding profiles dynamically.
*   **Observability & Cost Tracking**: Built-in Langfuse integration to seamlessly track LLM token usage, latencies, and generation costs across sessions.
*   **Sleek Streamlit Dashboard**: Dark-mode optimized UI featuring candidate comparison metrics, radar/bar charts, and expandable parsed payload views.

## 🏗️ Architecture

1.  **Ingestion & Parsing**: Multi-layer parsing (`parsers/`) handles both PDF and DOCX uploads, intelligently re-ordering text blocks based on visual coordinates to reconstruct 2-column formats.
2.  **Extraction Pipeline**:
    *   **Regex Engine**: Grabs exact pattern matches (contact info, profile URLs) and applies section-aware experience approximations.
    *   **LLM Engine**: Extracts structured JSON data according to highly specific, hallucination-resistant prompts (`extractors/`).
3.  **Enrichment**: Analyzers (`analyzers/`) hydrate profiles with open-source and competitive programming statistics.
4.  **Match & Rank Engine**: SentenceTransformer embeddings compute cosine similarity of parsed CV to the Job Description (JD). Next, the `ats_scorer.py` evaluates candidate tiers (fresher vs experienced).

## 🚀 Setup & Installation

### 1. Requirements

Ensure you have **Python 3.10+** installed.

```bash
git clone <your-repo-url>
cd AI_Resume_Parser

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

Open `.env` and configure your keys:
*   `OLLAMA_MODEL` / `OLLAMA_HOST`: Set to your running LLM endpoint (default `gpt-oss:120b-cloud` if using an Ollama cloud proxy, or a local `llama3`/`qwen` model).
*   `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`: Add keys if you wish to track your prompts and latencies.
*   `GITHUB_TOKEN`: Add a GitHub Personal Access Token to prevent API rate limiting.

### 3. Running the App

Start the Streamlit development server:

```bash
streamlit run app.py
```

Navigate to the Local URL provided in the terminal (usually `http://localhost:8501`).

## ⚙️ Configuration Details

| Missing Configurations | Notes |
| :--- | :--- |
| **Missing LLM Model?** | The system currently defaults to using `ollama`. If running locally without Docker, ensure `ollama serve` is active via CLI. |
| **False-positives on Dates?** | Our system implements a localized extraction engine ensuring only *working dates* residing below an *Experience* header are calculated. It avoids mischaracterizing continuous college enrollment dates as job history. |

## 📦 Project Structure

```text
AI_Resume_Parser/
├── .env.example              # Template environment
├── .gitignore                # Ignored file patterns
├── .streamlit/               # Streamlit styling parameters
│   └── config.toml
├── analyzers/                # 3rd-party profile enrichment integrations
│   ├── github_analyzer.py
│   └── leetcode_analyzer.py
├── extractors/               # Information retrieval nodes
│   ├── llm_extractor.py
│   └── regex_extractor.py
├── parsers/                  # Intelligent ingestion modules
│   ├── docx_parser.py
│   └── pdf_parser.py
├── scoring/                  # Matching algorithms
│   ├── ats_scorer.py
│   └── embedding_matcher.py
├── utils/                    # Global utilities
│   ├── langfuse_tracker.py
│   └── logger.py
├── app.py                    # Main Streamlit executable
└── requirements.txt          # Package dependencies
```

## 🛡️ License

MIT License. See `LICENSE` for more information.

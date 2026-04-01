# Resume Engine AI (v2)

A production-grade, Streamlit-based AI resume parsing and evaluation engine. It extracts structured data from complex resume layouts, enriches candidates with GitHub and LeetCode analytics, and provides deep scoring with confidence assessments -- all evaluated against a target Job Description.

**Version 2** introduces experience-level routing, consistency checking, claim validation, credibility scoring, and a full recruiter toolkit.

---

## What's New in v2

| Feature | Description |
|---|---|
| **Dynamic Level Routing** | Candidates are classified as Junior / Mid / Senior / Lead, each receiving level-specific evaluation weights |
| **Consistency Checker** | Timeline validation, skill-experience alignment, metric realism checks, AI content detection, gap analysis |
| **Claim Validator** | Per-skill evidence mapping, buzzword detection, career-switcher detection, depth-mismatch flagging |
| **Confidence Scoring** | Every candidate receives one of: `strong_candidate`, `review_recommended`, `proceed_with_caution`, `flag_for_review` |
| **Proof-of-Work Evaluator** | Deep GitHub analysis for freshers -- commit patterns, documentation quality, tech stack complexity |
| **Business Impact Evaluator** | Metric density analysis and quantifiable achievement verification for experienced candidates |
| **Recruiter Toolkit** | Four new panels: Timeline & Growth, Auto Email Generator, Skill Ontology Explorer, Candidate Deep Dive |
| **OpenAI Support** | Selectable LLM provider -- Ollama (local/cloud) or OpenAI GPT-3.5-turbo via `LLM_PROVIDER` env var |
| **Enhanced GitHub Analysis** | Commit pattern analysis, code quality signals, repo quality scoring, tech complexity assessment |

---

## Table of Contents

- [Architecture](#architecture)
- [Core Pipeline](#core-pipeline)
- [v2 Framework Details](#v2-framework-details)
- [Scoring Weights](#scoring-weights)
- [UI Tabs](#ui-tabs)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [License](#license)

---

## Architecture

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| UI | Streamlit with Plotly charts |
| LLM | Ollama (local or cloud) **or** OpenAI GPT-3.5-turbo |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Observability | Langfuse v2 |

---

## Core Pipeline

Each uploaded resume passes through a 12-step pipeline:

| Step | Module | Description |
|---:|---|---|
| 1 | **Document Parsing** (`parsers/`) | Multi-stage PDF extraction (PyMuPDF -> pdfplumber -> pdfminer) with column-aware layout detection. DOCX support included. |
| 2 | **Regex Extraction** (`extractors/regex_extractor.py`) | Fast rule-based extraction: email, phone, GitHub/LinkedIn/LeetCode URLs, 87+ skill keywords, section-aware experience calculation, student detection. |
| 3 | **LLM Extraction** (`extractors/llm_extractor.py`) | Semantic extraction of structured fields: name, skills, work experience, projects, education, candidate type. |
| 4 | **GitHub Enrichment** (`analyzers/github_analyzer.py`) | REST API analysis: repos, stars, forks, languages, followers, activity score. v2 adds commit patterns, code quality signals, repo quality, and tech complexity. |
| 5 | **LeetCode Enrichment** (`analyzers/leetcode_analyzer.py`) | GraphQL API analysis: problem counts, difficulty distribution, ranking, badges. |
| 6 | **Embedding Similarity** (`scoring/embedding_matcher.py`) | Cosine similarity between the full resume text and the Job Description using `all-MiniLM-L6-v2`. |
| 7 | **Fuzzy Skill Match** (`scoring/embedding_matcher.py`) | Per-skill embedding comparison with a 0.70 similarity threshold. |
| 8 | **ATS Scoring** (`scoring/ats_scorer.py`) | Weighted composite score (0--100) with fresher-aware weight adjustments. |
| 9 | **Consistency Check** [v2] (`analyzers/consistency_checker.py`) | Timeline validation, skill-experience alignment, metric realism, AI content detection, gap analysis. |
| 10 | **Claim Validation** [v2] (`analyzers/claim_validator.py`) | Skill evidence mapping, buzzword detection, career-switcher detection, depth-mismatch flagging, bonus signal recognition. |
| 11 | **Level Classification** [v2] (`scoring/level_router.py`) | Junior / Mid / Senior / Lead routing with level-specific evaluation weights. |
| 12 | **Confidence Scoring** [v2] (`scoring/ats_scorer.py`) | Overall confidence assessment combining ATS score, consistency results, and credibility signals. |

---

## v2 Framework Details

### Semantic Skill Matching

Goes beyond keyword matching. Uses sentence-transformer embeddings to verify whether a candidate's skill usage context genuinely matches the JD requirements, reducing false positives from keyword stuffing.

### Dynamic Routing by Experience Level

Candidates are classified into one of four levels: **Junior**, **Mid**, **Senior**, or **Lead**. Each level receives a different set of evaluation weights, reflecting what matters most at that career stage:

- **Junior** -- Heavy emphasis on projects and skills (proof-of-work path).
- **Mid** -- Balanced between proof-of-work and business impact.
- **Senior** -- Heavy emphasis on experience depth and JD alignment.
- **Lead** -- Dominated by experience and leadership signals.

### Proof-of-Work Evaluator (Freshers)

For candidates without significant professional experience, the system performs deep GitHub analysis:

- Commit patterns (iterative development vs. code dumps)
- Documentation quality across repositories
- Tech stack complexity and diversity
- Contribution consistency over time
- Project evaluation for candidates without GitHub profiles

### Business Impact Evaluator (Experienced)

For mid-level and above candidates:

- Metric density analysis across work experience entries
- Buzzword vs. fact detection
- Quantifiable achievement verification

### Consistency and Red Flag Detection

Automated checks for resume inconsistencies:

- **Timeline impossibilities** -- Graduating in 2024 but claiming 10 years of experience.
- **Skill padding** -- Listing 15+ skills with zero supporting projects.
- **Inflated metrics** -- Claims of >99% improvement without context.
- **AI-generated content** -- Risk scoring for synthetically produced resume text.

### Claim Validation and Credibility

Per-skill evidence mapping that connects claimed skills to demonstrated usage:

- Career-switcher detection with appropriate context adjustments.
- Depth mismatch flagging (e.g., a student claiming senior-level skill proficiency).
- Bonus signal recognition: certifications, open source contributions, publications.

### Confidence Scoring

Every candidate receives one of four confidence labels:

| Label | Meaning |
|---|---|
| `strong_candidate` | High ATS score, consistent resume, strong credibility signals |
| `review_recommended` | Solid profile with minor gaps worth a recruiter's attention |
| `proceed_with_caution` | Notable inconsistencies or unverified claims detected |
| `flag_for_review` | Significant red flags requiring manual review |

---

## Scoring Weights

| Signal | Standard | Fresher | Junior (v2) | Mid (v2) | Senior (v2) | Lead (v2) |
|---|---|---|---|---|---|---|
| JD Similarity | 0.40 | 0.40 | 0.35 | 0.35 | 0.40 | 0.40 |
| Skill Match | 0.25 | 0.30 | 0.25 | 0.25 | 0.20 | 0.15 |
| Projects | 0.15 | 0.20 | 0.25 | 0.15 | 0.10 | 0.05 |
| GitHub | 0.10 | 0.05 | 0.08 | 0.05 | 0.05 | 0.00 |
| LeetCode | 0.05 | 0.05 | 0.07 | 0.05 | 0.00 | 0.00 |
| Experience | 0.05 | 0.00 | 0.00 | 0.15 | 0.25 | 0.40 |

---

## UI Tabs

### 1. Job Description

Paste a raw Job Description and parse it via the LLM into structured fields (required skills, experience level, responsibilities).

### 2. Upload Resumes

Upload one or more PDF/DOCX files. Each resume runs through the full 12-step pipeline with real-time progress feedback.

### 3. Rankings

- ATS score bar chart (Plotly)
- Skills radar visualization
- Confidence overview across all candidates
- Detailed ranking cards with score breakdowns

### 4. Compare

Side-by-side comparison of any two candidates. Includes an LLM-generated explanation of relative strengths and weaknesses.

### 5. Recruiter Tools

Four dedicated panels:

| Panel | Purpose |
|---|---|
| **Timeline and Growth Analysis** | Visualize career progression, job tenure, and growth trajectory |
| **Auto Email Generator** | Generate shortlist, rejection, or interview invite emails using the LLM |
| **Skill Ontology Explorer** | Domain mapping, role fit assessment, skill gap identification, tech currency analysis |
| **Candidate Deep Dive** | Consistency report, claim evidence map, red flags summary, hiring recommendation |

---

## Project Structure

```
Resume_Pasrser_Engine_POC/
├── app.py                          # Streamlit orchestrator (entry point)
├── requirements.txt
├── .env.example
├── .streamlit/config.toml
│
├── parsers/
│   ├── pdf_parser.py               # Multi-stage PDF extraction
│   └── docx_parser.py              # DOCX extraction
│
├── extractors/
│   ├── llm_extractor.py            # LLM-based semantic extraction
│   ├── regex_extractor.py          # Rule-based field extraction
│   └── email_generator.py          # Recruiter email generation
│
├── analyzers/
│   ├── github_analyzer.py          # GitHub profile analysis + v2 deep signals
│   ├── leetcode_analyzer.py        # LeetCode stats analysis
│   ├── skill_ontology.py           # Skill mapping + v2 tech currency
│   ├── timeline_analyzer.py        # Career timeline + growth score
│   ├── consistency_checker.py      # [v2] Resume consistency + red flags
│   └── claim_validator.py          # [v2] Claim validation + credibility
│
├── scoring/
│   ├── ats_scorer.py               # ATS composite score + v2 confidence
│   ├── embedding_matcher.py        # Sentence-transformer matching
│   └── level_router.py             # [v2] Level classification + routing
│
├── ui/
│   ├── components.py               # Charts and visual components
│   ├── sidebar.py                  # Sidebar controls
│   ├── tab_jd.py                   # Job Description tab
│   ├── tab_upload.py               # Upload + processing pipeline
│   ├── tab_rankings.py             # Candidate rankings
│   ├── tab_compare.py              # Side-by-side comparison
│   ├── tab_recruiter.py            # Recruiter tools (4 panels)
│   ├── tab_obs.py                  # Observability (Langfuse)
│   ├── tab_logs.py                 # Processing logs
│   └── utils.py                    # Shared utilities
│
└── utils/
    ├── logger.py                   # Logging configuration
    └── langfuse_tracker.py         # Langfuse integration
```

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd Resume_Pasrser_Engine_POC

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
```

Edit `.env` and set the required values for your chosen LLM provider:

- **Ollama (default):** Set `OLLAMA_API_KEY` if using Ollama Cloud, or leave blank for a local instance.
- **OpenAI:** Set `LLM_PROVIDER=openai` and provide your `OPENAI_API_KEY`.
- **Optional:** Add `GITHUB_TOKEN` for higher API rate limits (5000 requests/hour) and `LANGFUSE_*` keys for observability.

### 3. Run

```bash
streamlit run app.py
```

The application will open in your browser (default: `http://localhost:8501`).

---

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `LLM_PROVIDER` | No | `ollama` | LLM backend: `"ollama"` or `"openai"` |
| `OLLAMA_API_KEY` | For Ollama Cloud | `""` | Ollama Cloud bearer token |
| `OLLAMA_HOST` | No | auto-detected | Ollama server URL |
| `OLLAMA_MODEL` | No | `gpt-oss:120b-cloud` | Ollama model name |
| `OPENAI_API_KEY` | For OpenAI | `""` | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-3.5-turbo` | OpenAI model name |
| `LANGFUSE_PUBLIC_KEY` | No | `""` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | No | `""` | Langfuse secret key |
| `LANGFUSE_HOST` | No | `https://cloud.langfuse.com` | Langfuse server URL |
| `GITHUB_TOKEN` | No | `""` | GitHub Personal Access Token (5000 req/hr) |
| `COMPANY_NAME` | No | `"our company"` | Company name used in generated emails |
| `LOG_LEVEL` | No | `INFO` | Python logging level |

---

## License

MIT

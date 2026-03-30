"""
extractors/regex_extractor.py  –  Fast regex-based field extraction

Key rules:
- NEVER compute experience from education/project date ranges
- Section-aware: only extract work dates from EXPERIENCE sections
- Separate full-time vs internship
- Student detection: present ongoing degree = is_student=True
"""
import re
from typing import Any, Dict, List, Tuple
from utils.logger import get_logger

log = get_logger("regex_extractor")

# ── Email ──────────────────────────────────────────────────────────────────────
EMAIL_RE = re.compile(
    r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}',
    re.I,
)

# ── Phone ──────────────────────────────────────────────────────────────────────
PHONE_RE = re.compile(
    r'(?<!\d)(?:\+?\d{1,3}[\s\-.]?)?(?:\(?\d{2,4}\)?[\s\-.]?)?\d{3,4}[\s\-.]?\d{4}(?!\d)'
)

# ── Social profiles ────────────────────────────────────────────────────────────
GITHUB_RE   = re.compile(r'(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_\-\.]+)',   re.I)
LINKEDIN_RE = re.compile(r'(?:https?://)?(?:www\.)?linkedin\.com/in/([A-Za-z0-9_\-\.%]+)', re.I)
LEETCODE_RE = re.compile(r'(?:https?://)?(?:www\.)?leetcode\.com/(?:u/)?([A-Za-z0-9_\-\.]+)', re.I)
URL_RE      = re.compile(r'https?://[^\s\)\]>,"\';]+', re.I)

# ── Education ──────────────────────────────────────────────────────────────────
DEGREE_RE = re.compile(
    r'\b(B\.?Tech|B\.?E\.?|B\.?Sc\.?|B\.?S\.?|M\.?Tech|M\.?Sc\.?|M\.?S\.?|'
    r'MBA|MCA|BCA|Ph\.?D|Bachelor|Master|Doctor|BE|ME|B\.Eng|M\.Eng)\b',
    re.I,
)

# ── Known tech skills ─────────────────────────────────────────────────────────
SKILL_KEYWORDS = {
    # Core languages
    "Python","Java","C","C++","C/C++","C#","JavaScript","TypeScript","Go","Golang","Rust",
    "Kotlin","Swift","Ruby","PHP","Scala","R","MATLAB","Bash","Shell","Perl","Lua",
    # Database / query
    "SQL","MySQL","PostgreSQL","SQLite","MS SQL","Oracle DB","PL/SQL","T-SQL",
    "MongoDB","Redis","Cassandra","DynamoDB","BigQuery","Snowflake","Supabase",
    "Elasticsearch","Neo4j","Pinecone","ChromaDB","Weaviate","Qdrant",
    # Web frameworks
    "React","Angular","Vue","Node.js","Django","FastAPI","Flask","Spring",
    "Spring Boot","Express","Next.js","Nuxt","Svelte","Tailwind","Bootstrap",
    "jQuery","WordPress","Gatsby","Remix","Astro",
    # APIs & protocols
    "REST API","RESTful APIs","REST","GraphQL","gRPC","WebSocket","SOAP",
    "OAuth","JWT","OpenAPI","Swagger","Postman","API Design",
    # AI / ML / DL
    "TensorFlow","PyTorch","scikit-learn","Keras","XGBoost","LightGBM",
    "Hugging Face","LangChain","LangGraph","LlamaIndex","OpenAI","Gemini AI",
    "Google Gemini AI","Anthropic","Mistral","Ollama","Groq",
    "Machine Learning","Deep Learning","NLP","Computer Vision","MLOps",
    "RAG","Vector Database","Embeddings","Fine-tuning","RLHF",
    "Stable Diffusion","BERT","GPT","Transformer","YOLO","OpenCV",
    # DevOps / Cloud
    "Docker","Kubernetes","AWS","GCP","Azure","Terraform","Ansible",
    "CI/CD","GitHub Actions","Jenkins","ArgoCD","Helm","Linux","Nginx",
    "Prometheus","Grafana","ELK Stack","Datadog","Vercel","Netlify",
    # Data engineering
    "Kafka","Spark","Hadoop","Airflow","dbt","Flink","Pandas","NumPy",
    "Power BI","Tableau","Looker","ETL",
    # Networking / SDN
    "Ryu","Mininet","SDN","OpenFlow","Wireshark","TCP/IP","DNS",
    "OSPF","BGP","VLAN","Firewall","VPN","NAT",
    # Automation / RPA / Bots
    "N8N","Zapier","Make","Selenium","Playwright","Puppeteer",
    "Telegram Bot","Discord Bot","Slack Bot","WhatsApp Bot",
    # Google ecosystem
    "Gmail API","Google Sheets API","Google Drive API","Google Cloud",
    "Firebase","Google Maps API","Google Analytics","Google Workspace",
    # Version control & collaboration
    "Git","GitHub","GitLab","Bitbucket","Jira","Confluence","Notion",
    # Design & product
    "Figma","Adobe XD","Canva","Sketch",
    # Practices
    "Agile","Scrum","Kanban","TDD","BDD","Microservices","System Design",
    "OOP","Functional Programming","Data Structures","Algorithms",
}

# ── Skill inference map ────────────────────────────────────────────────────────
SKILL_INFERENCE_MAP: Dict[str, List[str]] = {
    "rag":            ["NLP","Vector Database","Embeddings"],
    "langgraph":      ["LLM Orchestration","Python","LangChain"],
    "langchain":      ["Python","NLP","LLM"],
    "llama":          ["Machine Learning","NLP","Python"],
    "gpt":            ["Machine Learning","NLP","OpenAI"],
    "gemini":         ["Machine Learning","NLP","Gemini AI"],
    "fastapi":        ["Python","REST API","API Design"],
    "transformer":    ["Deep Learning","NLP","PyTorch"],
    "k8s":            ["Kubernetes","Docker"],
    "embeddings":     ["NLP","Vector Database","Machine Learning"],
    "vector":         ["Vector Database","Embeddings"],
    "fine-tun":       ["Deep Learning","PyTorch","Machine Learning"],
    "computer vision":["Deep Learning","PyTorch","OpenCV"],
    "microservice":   ["Docker","REST API","API Design"],
    "sdn":            ["Ryu","Mininet","OpenFlow"],
    "openflow":       ["SDN","Ryu","Mininet"],
    "restful":        ["REST API","API Design"],
    "n8n":            ["Automation","N8N"],
    "telegram":       ["Telegram Bot","Python"],
    "gmail api":      ["Gmail API","Google Workspace"],
    "google sheets":  ["Google Sheets API","Google Workspace"],
    "c/c++":          ["C","C++"],
}

# ── Section heading patterns ──────────────────────────────────────────────────
EXP_SECTION_RE = re.compile(
    r'(?:^|\n)\s*(?:'
    r'(?:work\s+)?experience|employment|professional\s+experience|career|'
    r'work\s+history|positions?\s+held|internship'
    r')\s*\n',
    re.I | re.M,
)

EDU_SECTION_RE = re.compile(
    r'(?:^|\n)\s*(?:education|academic|qualification|degree)\s*\n',
    re.I | re.M,
)

PROJ_SECTION_RE = re.compile(
    r'(?:^|\n)\s*(?:projects?|personal\s+projects?|academic\s+projects?|portfolio)\s*\n',
    re.I | re.M,
)

OTHER_SECTIONS_RE = re.compile(
    r'(?:^|\n)\s*(?:skills?|certifications?|achievements?|awards?|'
    r'publications?|languages?|interests?|volunteering)\s*\n',
    re.I | re.M,
)

# ── Date patterns used in experience sections only ────────────────────────────
# Matches: May 2022, 05/2022, 2022, Present, Current
DATE_RE = re.compile(
    r'(?:'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}'  # Month Year
    r'|(?:\d{1,2}[/\-]\d{4})'         # MM/YYYY
    r'|\d{4}'                           # YYYY
    r'|Present|Current|Now|Ongoing'
    r')',
    re.I,
)

DATE_RANGE_RE = re.compile(
    r'(?:'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}'
    r'|(?:\d{1,2}[/\-]\d{4})'
    r'|\d{4}'
    r')'
    r'\s*(?:\u2013|\u2014|-|to)\s*'
    r'(?:'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}'
    r'|(?:\d{1,2}[/\-]\d{4})'
    r'|\d{4}'
    r'|Present|Current|Now|Ongoing'
    r')',
    re.I,
)

INTERNSHIP_TITLE_RE = re.compile(
    r'\b(?:intern|trainee|apprentice|co-op|contract)\b', re.I
)

STUDENT_DEGREE_ONGOING_RE = re.compile(
    r'(?:B\.?Tech|B\.?E\.?|B\.?Sc|B\.?S\.?|M\.?Tech|M\.?Sc|Bachelor|Master)'
    r'.*?(?:20\d\d\s*[-\u2013\u2014]\s*(?:Present|Current|20\d\d))',
    re.I | re.S,
)

CURRENT_YEAR = 2026


def _parse_year(s: str) -> int:
    """Extract a 4-digit year from a date string. Returns 0 if not found."""
    m = re.search(r'\b(20\d\d|19\d\d)\b', s)
    return int(m.group(1)) if m else 0


def _is_present(s: str) -> bool:
    return bool(re.search(r'present|current|now|ongoing', s, re.I))


def _duration_months(start_str: str, end_str: str) -> float:
    """Estimate duration in months between two date strings."""
    sy = _parse_year(start_str)
    if _is_present(end_str):
        ey = CURRENT_YEAR
    else:
        ey = _parse_year(end_str)

    if not sy or not ey or ey < sy:
        return 0.0

    # Try to extract month numbers
    month_map = {
        "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
        "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    }
    sm = 1
    em = 12
    for k, v in month_map.items():
        if k in start_str.lower():
            sm = v
        if k in end_str.lower():
            em = v
    # Check MM/YYYY format
    mm_re = re.compile(r'(\d{1,2})[/\-](\d{4})')
    ms = mm_re.search(start_str)
    me = mm_re.search(end_str)
    if ms:
        sm = int(ms.group(1))
        sy = int(ms.group(2))
    if me:
        em = int(me.group(1))
        ey = int(me.group(2))

    total = (ey - sy) * 12 + (em - sm)
    return max(0.0, float(total))


def _extract_experience_section(text: str) -> Tuple[str, str, str]:
    """
    Split the resume text into:
      (experience_section, education_section, other_text)
    using section heading positions.
    """
    # Find first experience section
    exp_match   = EXP_SECTION_RE.search(text)
    edu_match   = EDU_SECTION_RE.search(text)
    proj_match  = PROJ_SECTION_RE.search(text)
    other_match = OTHER_SECTIONS_RE.search(text)

    # All next-section starts after the experience section
    if not exp_match:
        return "", text, text

    exp_start = exp_match.start()

    # Find the end of the experience section (start of next major section)
    boundaries = [
        m.start() for m in [edu_match, proj_match, other_match]
        if m and m.start() > exp_start
    ]
    exp_end = min(boundaries) if boundaries else len(text)
    exp_section = text[exp_start:exp_end]

    edu_section = text[edu_match.start():] if edu_match else ""
    return exp_section, edu_section, text


# ── Role/title heuristics for experience line detection ───────────────────────
_ROLE_TITLE_RE = re.compile(
    r'\b(?:intern|engineer|developer|analyst|architect|lead|manager|scientist|'
    r'designer|consultant|specialist|coordinator|director|head|associate|'
    r'trainee|apprentice|co-op|officer|executive|administrator|technician|'
    r'programmer|researcher|team\s+leader)\b',
    re.I,
)


def _split_role_company(line: str) -> Tuple[str, str]:
    """
    Handle patterns like:
      "Full Stack Developer – Intern – UpKraft Technologies"
      "IT Technical Intern & Team Leader @ Fifty Is Nifty"
      "Software Engineer, Google"
    """
    # Try em/en dash or multiple dashes first
    for sep in [r'\u2013', r'\u2014', r'\s+[-–—]+\s+', r'@', r',\s+']:
        parts = re.split(sep, line, maxsplit=1)
        if len(parts) == 2:
            role = parts[0].strip()
            company = parts[1].strip()
            # If role still has intern/company signal at end, strip it
            return role, company
    return line.strip(), ""


def _extract_date_range_from_lines(lines: List[str], start: int, lookahead: int = 6) -> str:
    """Scan up to `lookahead` lines after `start` looking for a date range."""
    for i in range(start, min(start + lookahead, len(lines))):
        m = DATE_RANGE_RE.search(lines[i])
        if m:
            return m.group(0)
    return ""


def _compute_experience_from_section(exp_text: str) -> Tuple[float, float]:
    """
    3-pass robust experience extractor:

    Pass 1 – Block-based (standard resumes, date inline)
    Pass 2 – Line-scanning (non-standard: date at bottom, dashes, etc.)
    Pass 3 – Fallback: just count any date range in section with intern check

    Returns (full_time_years, internship_months).
    """
    full_time_months = 0.0
    intern_months    = 0.0
    seen_ranges: set = set()  # avoid double-counting

    lines = exp_text.splitlines()

    # ── Pass 1: block-based ────────────────────────────────────────────────────
    blocks = re.split(r'\n\s*\n', exp_text.strip())
    for block in blocks:
        ranges = DATE_RANGE_RE.findall(block)
        if not ranges:
            continue
        is_intern = bool(INTERNSHIP_TITLE_RE.search(block))
        for rng in ranges:
            parts = re.split(r'\s*(?:\u2013|\u2014|-{1,2}|to)\s*', rng, maxsplit=1)
            if len(parts) != 2:
                continue
            dur = _duration_months(parts[0], parts[1])
            if dur <= 0:
                continue
            key = rng.strip().lower()
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            if is_intern:
                intern_months += dur
            else:
                full_time_months += dur

    # ── Pass 2: line-scanning for non-standard layouts ─────────────────────────
    # Handles: "Role – Company\n...description...\nMM/YYYY - MM/YYYY"
    i = 0
    while i < len(lines):
        line = lines[i]
        # If this line looks like a job title/role
        if _ROLE_TITLE_RE.search(line) and len(line) < 120:
            is_intern = bool(INTERNSHIP_TITLE_RE.search(line))
            # Search for a date range in surrounding lines
            rng = _extract_date_range_from_lines(lines, i, lookahead=7)
            if rng:
                key = rng.strip().lower()
                if key not in seen_ranges:
                    seen_ranges.add(key)
                    parts = re.split(r'\s*(?:\u2013|\u2014|-{1,2}|to)\s*', rng, maxsplit=1)
                    if len(parts) == 2:
                        dur = _duration_months(parts[0], parts[1])
                        if dur > 0:
                            if is_intern:
                                intern_months += dur
                            else:
                                full_time_months += dur
        i += 1

    # ── Pass 3: bare date range fallback ──────────────────────────────────────
    # If still nothing found, grab ALL date ranges in section
    if full_time_months == 0 and intern_months == 0:
        all_ranges = DATE_RANGE_RE.findall(exp_text)
        is_intern_ctx = bool(INTERNSHIP_TITLE_RE.search(exp_text))
        for rng in all_ranges:
            key = rng.strip().lower()
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            parts = re.split(r'\s*(?:\u2013|\u2014|-{1,2}|to)\s*', rng, maxsplit=1)
            if len(parts) != 2:
                continue
            dur = _duration_months(parts[0], parts[1])
            if dur > 0:
                if is_intern_ctx:
                    intern_months += dur
                else:
                    full_time_months += dur

    full_time_years = round(full_time_months / 12.0, 1)
    return full_time_years, intern_months


def _is_student(text: str) -> bool:
    """True if an ongoing undergrad/grad degree is mentioned."""
    return bool(STUDENT_DEGREE_ONGOING_RE.search(text)) or (
        bool(DEGREE_RE.search(text)) and
        bool(re.search(r'(?:20\d\d\s*[-\u2013]\s*(?:Present|2[05-9]\d\d)|Present)', text, re.I))
    )


def extract_regex_fields(text: str) -> Dict[str, Any]:
    """
    Extract cheap fields from raw resume text.
    Returns properly calculated experience fields.
    """
    log.info("Running regex extraction (%d chars)", len(text))

    # ── Social ─────────────────────────────────────────────────────────────────
    emails  = EMAIL_RE.findall(text)
    phones  = PHONE_RE.findall(text)

    github   = GITHUB_RE.search(text)
    linkedin = LINKEDIN_RE.search(text)
    leetcode = LEETCODE_RE.search(text)

    gh_user = github.group(1) if github else None
    if gh_user and gh_user.lower() in {"login","signup","joins","apps","features"}:
        gh_user = None

    # ── Education ──────────────────────────────────────────────────────────────
    degrees = list({d for d in DEGREE_RE.findall(text)})

    # ── Student detection ──────────────────────────────────────────────────────
    student = _is_student(text)

    # ── Section-aware experience ───────────────────────────────────────────────
    exp_section, edu_section, _ = _extract_experience_section(text)

    if student:
        # Students have no full-time experience by definition
        full_time_years  = 0.0
        _, intern_months = _compute_experience_from_section(exp_section)
    else:
        full_time_years, intern_months = _compute_experience_from_section(exp_section)

    intern_months = min(intern_months, 36.0)  # cap sanity: max 3yr internship
    intern_years_weighted = round(intern_months / 12.0 * 1.0, 2)  # 1.0 weight (same as full-time)
    effective_exp = round(full_time_years + intern_years_weighted, 2)

    candidate_type = (
        "student"     if student
        else "fresher"     if full_time_years < 1
        else "experienced"
    )

    # ── Skills ─────────────────────────────────────────────────────────────────
    text_lower = text.lower()
    found_skills = sorted({s for s in SKILL_KEYWORDS if s.lower() in text_lower})
    inferred: List[str] = []
    for kw, extras in SKILL_INFERENCE_MAP.items():
        if kw in text_lower:
            inferred.extend(extras)
    all_skills = sorted(set(found_skills + inferred))

    result: Dict[str, Any] = {
        "email":                    emails[0] if emails else None,
        "phone":                    phones[0] if phones else None,
        "github_url":               f"https://github.com/{gh_user}" if gh_user else None,
        "github_username":          gh_user,
        "linkedin_url":             (f"https://linkedin.com/in/{linkedin.group(1)}" if linkedin else None),
        "leetcode_url":             (f"https://leetcode.com/{leetcode.group(1)}" if leetcode else None),
        "leetcode_username":        leetcode.group(1) if leetcode else None,
        "degrees":                  degrees,
        "tech_skills":              all_skills,
        "inferred_skills":          inferred,
        # ── experience fields ──────────────────────────────────────────────────
        "full_time_experience_years":  full_time_years,
        "internship_months":           intern_months,
        "intern_years_weighted":       intern_years_weighted,
        "effective_experience_years":  effective_exp,
        "is_student":                  student,
        "candidate_type":              candidate_type,
        # legacy key (used downstream)
        "exp_years_estimate":          effective_exp,
        "all_urls":                    URL_RE.findall(text),
    }

    log.info(
        "Regex done → type=%s ft=%gy intern=%gmo eff=%gy skills=%d",
        candidate_type, full_time_years,
        intern_months, effective_exp, len(all_skills),
    )
    return result

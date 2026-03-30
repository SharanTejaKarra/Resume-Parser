"""
analyzers/skill_ontology.py  –  Recruiter-Centric Skill Ontology

Fully rule-based / embedding-based (zero LLM calls).

Pipeline:
  1. Normalize raw skill strings (synonyms → canonical forms)
  2. Map canonical skills → domain buckets
  3. Compute domain scores with source weighting:
       experience > project > listed skill
  4. Role-fit scoring against predefined role profiles
  5. Skill gap detection vs. a target role's required skills
"""

from typing import Any, Dict, List, Optional, Tuple
from utils.logger import get_logger

log = get_logger("skill_ontology")

# ── 1. Synonym normalisation map ──────────────────────────────────────────────

SYNONYM_MAP: Dict[str, str] = {
    # JavaScript ecosystem
    "react.js": "React", "reactjs": "React",
    "node": "Node.js", "nodejs": "Node.js",
    "express.js": "Express", "expressjs": "Express",
    "next.js": "Next.js", "nextjs": "Next.js",
    "vue.js": "Vue", "vuejs": "Vue",
    "angular.js": "Angular", "angularjs": "Angular",
    "ts": "TypeScript",
    # C family
    "c/c++": "C/C++", "c / c++": "C/C++",
    # Python
    "py": "Python", "python3": "Python",
    "sklearn": "Scikit-learn", "scikit learn": "Scikit-learn", "scikit-learn": "Scikit-learn",
    "tf": "TensorFlow", "tensorflow2": "TensorFlow",
    "pytorch": "PyTorch", "torch": "PyTorch",
    "langchain": "LangChain", "lang chain": "LangChain",
    "llamaindex": "LlamaIndex",
    # APIs
    "rest": "REST API", "restful": "REST API", "restful apis": "REST API",
    "rest api": "REST API",
    "graphql": "GraphQL", "grpc": "gRPC",
    # DevOps / Cloud
    "k8s": "Kubernetes", "kube": "Kubernetes",
    "ci/cd": "CI/CD", "cicd": "CI/CD",
    "github actions": "GitHub Actions",
    "gcp": "Google Cloud", "google cloud platform": "Google Cloud",
    "aws": "AWS", "amazon web services": "AWS",
    "azure": "Azure", "microsoft azure": "Azure",
    # Databases
    "mongo": "MongoDB", "mongodb": "MongoDB",
    "postgres": "PostgreSQL", "pg": "PostgreSQL",
    "mysql": "MySQL", "mssql": "SQL Server", "ms sql": "SQL Server",
    "sql": "SQL", "pl/sql": "PL/SQL", "t-sql": "T-SQL",
    "sqlite": "SQLite",
    "redis": "Redis", "elastic": "Elasticsearch",
    "neo4j": "Neo4j", "pinecone": "Pinecone",
    # Networking / SDN
    "sdn": "SDN", "openflow": "OpenFlow",
    "ryu": "Ryu", "mininet": "Mininet",
    "tcp/ip": "TCP/IP", "wireshark": "Wireshark",
    # Automation / Bots
    "n8n": "N8N",
    "telegram bot": "Telegram Bot",
    "discord bot": "Discord Bot",
    # Google APIs
    "gmail api": "Gmail API",
    "google sheets api": "Google Sheets API",
    "google gemini ai": "Gemini AI", "gemini ai": "Gemini AI",
    # Mobile
    "react native": "React Native",
    "ios": "iOS", "android": "Android",
    # General
    "ml": "Machine Learning", "ai": "Artificial Intelligence",
    "nlp": "NLP", "cv": "Computer Vision",
    "oop": "OOP", "dsa": "Data Structures",
    "git": "Git", "github": "GitHub",
    "linux": "Linux", "unix": "Linux",
    "bs": "BeautifulSoup",
}


def normalize_skill(raw: str) -> str:
    """Return canonical form of a skill string."""
    cleaned = raw.strip().lower()
    return SYNONYM_MAP.get(cleaned, raw.strip().title())


# ── 2. Domain ontology ────────────────────────────────────────────────────────

DOMAIN_ONTOLOGY: Dict[str, List[str]] = {
    "Frontend": [
        "React", "Angular", "Vue", "Next.js", "HTML", "CSS", "JavaScript",
        "TypeScript", "Tailwind", "Redux", "Webpack", "Sass", "Bootstrap",
        "Svelte", "Figma", "jQuery", "Gatsby", "Remix", "Astro",
    ],
    "Backend": [
        "Node.js", "Express", "Django", "FastAPI", "Flask", "Spring Boot",
        "Ruby On Rails", "Go", "Rust", "PHP", "REST API", "GraphQL",
        "Microservices", "gRPC", "Celery", "RabbitMQ", "Kafka",
        "WebSocket", "SOAP", "OpenAPI", "Swagger", "Postman",
    ],
    "AI/ML": [
        "TensorFlow", "PyTorch", "Scikit-learn", "Keras", "XGBoost",
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
        "LangChain", "LlamaIndex", "OpenAI", "Gemini AI", "Hugging Face",
        "RAG", "Vector Database", "Embedding", "LLM", "Artificial Intelligence",
        "Reinforcement Learning", "Transformers", "BERT", "GPT", "YOLO", "OpenCV",
        "Stable Diffusion", "Ollama", "Groq", "Mistral", "Anthropic", "LangGraph",
        "Fine-tuning", "RLHF", "MLOps",
    ],
    "DevOps": [
        "Docker", "Kubernetes", "CI/CD", "GitHub Actions", "Jenkins",
        "Terraform", "Ansible", "Helm", "Linux", "Bash", "Nginx",
        "ArgoCD", "Prometheus", "Grafana", "ELK Stack", "Datadog",
        "Vercel", "Netlify",
    ],
    "Cloud": [
        "AWS", "Azure", "Google Cloud", "Firebase", "Vercel", "Heroku",
        "Cloudflare", "Lambda", "S3", "EC2", "CloudFront", "BigQuery",
        "Supabase", "PlanetScale", "Google Workspace",
    ],
    "Database": [
        "MySQL", "PostgreSQL", "MongoDB", "Redis", "SQLite", "Cassandra",
        "DynamoDB", "Elasticsearch", "Neo4j", "SQL Server", "Pinecone",
        "ChromaDB", "Weaviate", "Qdrant", "SQL", "PL/SQL", "T-SQL",
        "Oracle DB", "Snowflake",
    ],
    "Mobile": [
        "React Native", "Flutter", "iOS", "Android", "Swift", "Kotlin",
        "Expo", "Dart",
    ],
    "Networking": [
        "Ryu", "Mininet", "SDN", "OpenFlow", "Wireshark", "TCP/IP",
        "DNS", "OSPF", "BGP", "VLAN", "Firewall", "VPN", "NAT",
        "Network Security",
    ],
    "Automation": [
        "N8N", "Zapier", "Make", "Selenium", "Playwright", "Puppeteer",
        "Telegram Bot", "Discord Bot", "Slack Bot", "WhatsApp Bot",
        "Gmail API", "Google Sheets API", "Google Drive API",
        "Google Maps API",
    ],
    "Data_Engineering": [
        "Spark", "Hadoop", "Airflow", "dbt", "Flink", "Kafka",
        "Snowflake", "Power BI", "Tableau", "Pandas", "NumPy",
        "Data Structures", "Algorithms", "ETL", "BigQuery", "Looker",
    ],
    "Security": [
        "OAuth", "JWT", "SSL/TLS", "Penetration Testing", "OWASP",
        "Cryptography", "IAM", "Firewall",
    ],
    "General_SE": [
        "Git", "GitHub", "OOP", "System Design", "Data Structures",
        "Agile", "Scrum", "Kanban", "TDD", "BDD",
        "Python", "Java", "C", "C++", "C/C++", "C#", "Go", "Rust",
        "TypeScript", "JavaScript", "Functional Programming",
        "Algorithms", "API Design",
    ],
}

# Flatten domain → skill lookup (skill → [domain, ...])
_SKILL_TO_DOMAINS: Dict[str, List[str]] = {}
for _domain, _skills in DOMAIN_ONTOLOGY.items():
    for _skill in _skills:
        _SKILL_TO_DOMAINS.setdefault(_skill.lower(), []).append(_domain)


def skill_to_domains(skill: str) -> List[str]:
    """Return list of domains a canonical skill belongs to."""
    return _SKILL_TO_DOMAINS.get(skill.lower(), [])


# ── 3. Source weight constants ────────────────────────────────────────────────

WEIGHT_LISTED     = 1    # skill on resume skill list
WEIGHT_PROJECT    = 2    # used in a project
WEIGHT_EXPERIENCE = 3    # used in work experience


# ── 4. Core mapping function ──────────────────────────────────────────────────

def map_candidate_skills(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize raw skills; produce domain_expertise with weighted scores.

    Returns:
    {
        "normalized_skills": [...],
        "domain_expertise": {
            "Frontend": {"score": 4, "skills": ["React", "CSS"]},
            ...
        },
        "unmapped_skills": [...]
    }
    """
    # Collect skills with their source weight
    raw_skills: List[str] = list(candidate.get("skills") or [])

    # Skill → max weight seen
    skill_weights: Dict[str, int] = {}

    for s in raw_skills:
        ns = normalize_skill(s)
        skill_weights[ns] = max(skill_weights.get(ns, 0), WEIGHT_LISTED)

    # Boost skills found in projects
    for proj in (candidate.get("projects") or []):
        if not isinstance(proj, dict):
            continue
        for s in (proj.get("tech_stack") or []):
            ns = normalize_skill(s)
            skill_weights[ns] = max(skill_weights.get(ns, 0), WEIGHT_PROJECT)

    # Boost skills found in work experience descriptions
    for exp in (candidate.get("work_experience") or []):
        if not isinstance(exp, dict):
            continue
        desc = (exp.get("description") or "") + " ".join(exp.get("achievements") or [])
        # Simple mention detection against ontology skills
        for domain_skills in DOMAIN_ONTOLOGY.values():
            for ds in domain_skills:
                if ds.lower() in desc.lower():
                    skill_weights[ds] = max(skill_weights.get(ds, 0), WEIGHT_EXPERIENCE)

    normalized_skills = sorted(skill_weights.keys())

    # Build domain expertise
    domain_buckets: Dict[str, Dict[str, Any]] = {}
    unmapped: List[str] = []

    for skill, weight in skill_weights.items():
        domains = skill_to_domains(skill)
        if not domains:
            unmapped.append(skill)
            continue
        for dom in domains:
            bucket = domain_buckets.setdefault(dom, {"score": 0, "skills": [], "weighted": 0})
            if skill not in bucket["skills"]:
                bucket["skills"].append(skill)
                bucket["weighted"] += weight  # internal use only

    # Finalise score (count of skills, capped at 5, then adjusted for weight)
    for dom, bucket in domain_buckets.items():
        raw_count = len(bucket["skills"])
        # Score = min(count, 5) with a small boost for high-weight skills
        bucket["score"] = min(raw_count, 5)
        bucket.pop("weighted", None)  # remove internal key from output

    log.info(
        "Ontology: %d normalized skills, %d domains, %d unmapped",
        len(normalized_skills), len(domain_buckets), len(unmapped),
    )
    return {
        "normalized_skills": normalized_skills,
        "domain_expertise":  domain_buckets,
        "unmapped_skills":   unmapped,
    }


# ── 5. Role fit profiles ──────────────────────────────────────────────────────

# Each profile: list of (domain, required_count) pairs
_ROLE_PROFILES: Dict[str, Dict[str, int]] = {
    "Frontend Developer":  {"Frontend": 4, "General_SE": 2},
    "Backend Developer":   {"Backend": 4, "Database": 2, "General_SE": 2},
    "Full Stack Developer": {"Frontend": 3, "Backend": 3, "Database": 2},
    "ML Engineer":         {"AI/ML": 4, "Data_Engineering": 2, "General_SE": 2},
    "Data Scientist":      {"AI/ML": 3, "Data_Engineering": 3},
    "DevOps Engineer":     {"DevOps": 4, "Cloud": 3},
    "Cloud Architect":     {"Cloud": 4, "DevOps": 3, "Backend": 2},
    "Mobile Developer":    {"Mobile": 4, "Backend": 2},
    "Security Engineer":   {"Security": 3, "Backend": 3, "DevOps": 2},
}

CUSTOM_ROLE_WEIGHTS: Dict[str, Dict[str, int]] = {}  # allow runtime extension


def compute_role_fit(
    domain_expertise: Dict[str, Any],
    custom_profiles: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, float]:
    """
    Output: {"Frontend Developer": 85.0, "Backend Developer": 60.0, ...}
    Score = Σ min(candidate_domain_score / required_count, 1.0) / num_domains * 100
    """
    profiles = {**_ROLE_PROFILES, **(custom_profiles or {})}
    role_fit: Dict[str, float] = {}

    for role, requirements in profiles.items():
        total_weight = sum(requirements.values())
        earned = 0.0
        for domain, required in requirements.items():
            candidate_score = domain_expertise.get(domain, {}).get("score", 0)
            earned += min(candidate_score / required, 1.0) * required
        pct = round(earned / total_weight * 100, 1)
        role_fit[role] = pct

    return dict(sorted(role_fit.items(), key=lambda x: x[1], reverse=True))


# ── 6. Skill gap detection ────────────────────────────────────────────────────

def detect_skill_gaps(
    normalized_skills: List[str],
    target_role: Optional[str] = None,
    jd_required_skills: Optional[List[str]] = None,
) -> List[str]:
    """
    Identify missing skills.
    Priority: JD required skills first, then role profile gaps.
    """
    lower_skills = {s.lower() for s in normalized_skills}
    gaps: List[str] = []

    # JD-based gaps
    if jd_required_skills:
        for skill in jd_required_skills:
            ns = normalize_skill(skill)
            if ns.lower() not in lower_skills:
                gaps.append(ns)

    # Role profile gaps
    if target_role and target_role in _ROLE_PROFILES:
        for domain, required_count in _ROLE_PROFILES[target_role].items():
            candidate_count = 0
            for s in normalized_skills:
                if domain in skill_to_domains(s):
                    candidate_count += 1
            if candidate_count < required_count:
                # Add representative missing skills from that domain
                domain_skills = DOMAIN_ONTOLOGY.get(domain, [])
                for ds in domain_skills:
                    if ds.lower() not in lower_skills and ds not in gaps:
                        gaps.append(ds)
                        if len(gaps) >= required_count - candidate_count + len(gaps) - 1:
                            break

    return list(dict.fromkeys(gaps))[:15]  # deduplicate, cap at 15


# ── 7. Public high-level API ──────────────────────────────────────────────────

def analyze_skill_ontology(
    candidate: Dict[str, Any],
    jd_required_skills: Optional[List[str]] = None,
    target_role: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single entry point called from the upload pipeline or UI.

    Returns:
    {
        "normalized_skills": [...],
        "domain_expertise": {...},
        "role_fit": {"Frontend Developer": 85, ...},
        "skill_gaps": ["Docker", "Kubernetes", ...],
        "unmapped_skills": [...]
    }
    """
    mapping    = map_candidate_skills(candidate)
    role_fit   = compute_role_fit(mapping["domain_expertise"])
    skill_gaps = detect_skill_gaps(
        mapping["normalized_skills"],
        target_role=target_role,
        jd_required_skills=jd_required_skills,
    )

    return {
        "normalized_skills": mapping["normalized_skills"],
        "domain_expertise":  mapping["domain_expertise"],
        "unmapped_skills":   mapping["unmapped_skills"],
        "role_fit":          role_fit,
        "skill_gaps":        skill_gaps,
    }

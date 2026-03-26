"""
scoring/embedding_matcher.py  –  Sentence-transformer embedding-based JD matching
Uses all-MiniLM-L6-v2 (fast, ~80 MB, no GPU needed).
"""
from typing import List, Tuple
import numpy as np
from utils.logger import get_logger

log = get_logger("embedding_matcher")

_model = None  # lazy-load singleton


def _load_model():
    global _model
    if _model is None:
        log.info("Loading sentence-transformer model (first run may take ~30s)…")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Model loaded ✓")
    return _model


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def embed_text(text: str) -> np.ndarray:
    """Return embedding vector for a text string (first 2000 chars)."""
    model = _load_model()
    truncated = text[:2000]
    vecs: np.ndarray = model.encode([truncated], convert_to_numpy=True)  # type: ignore[assignment]
    return vecs[0]


def compute_jd_similarity(resume_text: str, jd_text: str) -> float:
    """
    Compute cosine similarity between resume and JD embeddings.
    Returns float in [0, 1].
    """
    log.info("Computing embedding similarity…")
    rv = embed_text(resume_text)
    jv = embed_text(jd_text)
    sim = _cosine_sim(rv, jv)
    log.info("Embedding similarity: %.4f", sim)
    return sim


def compute_skill_match(
    resume_skills: List[str],
    jd_required:   List[str],
) -> Tuple[float, List[str], List[str]]:
    """
    Fuzzy skill match using embedding cosine similarity.
    Returns (score 0-100, matched_skills, missing_skills).
    """
    if not jd_required:
        return 0.0, [], []
    if not resume_skills:
        return 0.0, [], list(jd_required)

    model = _load_model()

    all_texts   = list(resume_skills) + list(jd_required)
    all_vectors: np.ndarray = model.encode(all_texts, convert_to_numpy=True)  # type: ignore[assignment]

    res_vecs = all_vectors[:len(resume_skills)]
    jd_vecs  = all_vectors[len(resume_skills):]

    THRESHOLD = 0.70
    matched: List[str] = []
    missing: List[str] = []

    for i, jd_skill in enumerate(jd_required):
        sims = [_cosine_sim(res_vecs[j], jd_vecs[i]) for j in range(len(resume_skills))]
        best_sim = max(sims) if sims else 0.0
        if best_sim >= THRESHOLD:
            matched.append(jd_skill)
        else:
            missing.append(jd_skill)

    score = round(float(len(matched)) / float(len(jd_required)) * 100.0, 1)
    log.info("Skill match: %d/%d = %.1f%%", len(matched), len(jd_required), score)
    return score, matched, missing

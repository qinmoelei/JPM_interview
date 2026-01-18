from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm.apiyi import extract_chat_content, request_chat_completion


OPINION_PATTERNS = {
    "qualified": re.compile(r"qualified opinion", re.I),
    "adverse": re.compile(r"adverse opinion", re.I),
    "disclaimer": re.compile(r"disclaimer of opinion", re.I),
    "going_concern": re.compile(r"material uncertainty.*going concern|going concern", re.I),
    "emphasis": re.compile(r"emphasis of matter", re.I),
}


RISK_KEYWORDS = [
    "liquidity",
    "going concern",
    "material weakness",
    "covenant",
    "default",
    "credit risk",
    "litigation",
    "restatement",
    "impairment",
    "contingency",
    "macro",
    "refinancing",
    "going-concern",
]


def _read_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        texts.append(text.replace("\x00", " ").strip())
    return "\n".join(texts)


def extract_audit_opinion(text: str) -> Dict[str, object]:
    flags = {name: bool(pattern.search(text)) for name, pattern in OPINION_PATTERNS.items()}
    if flags["adverse"]:
        opinion = "adverse"
    elif flags["disclaimer"]:
        opinion = "disclaimer"
    elif flags["qualified"]:
        opinion = "qualified"
    else:
        opinion = "unqualified"
    return {"opinion": opinion, "flags": flags}


def split_paragraphs(text: str, min_len: int = 80) -> List[str]:
    raw = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return [p for p in raw if len(p) >= min_len]


def rank_risk_paragraphs(text: str, top_k: int = 20) -> List[Dict[str, object]]:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    para_vecs = vectorizer.fit_transform(paragraphs)
    query_vec = vectorizer.transform([" ".join(RISK_KEYWORDS)])
    scores = cosine_similarity(para_vecs, query_vec).ravel()
    ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"rank": i + 1, "score": float(score), "paragraph": paragraphs[idx]} for i, (idx, score) in enumerate(ranked)]


async def llm_score_paragraphs(paragraphs: Sequence[str], model: Optional[str] = None) -> List[Dict[str, object]]:
    prompt = (
        "Score each paragraph for risk relevance on a 0-5 scale. "
        "Return JSON array of objects with fields: score, rationale."
    )
    content = prompt + "\n\n" + json.dumps(paragraphs, indent=2)
    response = await request_chat_completion(
        [{"role": "system", "content": "You are a risk analyst."}, {"role": "user", "content": content}],
        model=model,
        temperature=0.0,
        max_tokens=900,
    )
    raw = extract_chat_content(response)
    return json.loads(raw)


def analyze_pdf(pdf_path: Path, top_k: int = 20) -> Dict[str, object]:
    text = _read_pdf_text(pdf_path)
    opinion = extract_audit_opinion(text)
    ranked = rank_risk_paragraphs(text, top_k=top_k)
    return {
        "pdf": str(pdf_path),
        "opinion": opinion,
        "top_paragraphs": ranked,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

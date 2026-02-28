from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from rag_config import BRAND_TERMS, ROMAN_URDU_MARKERS

_URDU_BLOCK_RE = re.compile(r"[\u0600-\u06FF]")

@dataclass
class QueryLangProfile:
    original_query: str
    input_lang: str          # 'en' | 'ur' | 'roman_ur'
    query_en: str            # English form for domain classification + expansion
    output_lang: str         # 'en' | 'ur' | 'roman_ur'


def is_urdu_script(text: str) -> bool:
    return bool(_URDU_BLOCK_RE.search(text or ""))


def _safe_langdetect(text: str) -> str:
    try:
        import langdetect
        return langdetect.detect(text)
    except Exception:
        return "en"


def looks_like_roman_urdu(text: str) -> bool:
    if is_urdu_script(text):
        return False
    if re.search(r"[^\x00-\x7F]", text or ""):
        return False
    tokens = re.findall(r"[a-zA-Z']+", (text or "").lower())
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in ROMAN_URDU_MARKERS)
    return (hits >= 2) or (hits >= 1 and len(tokens) <= 6)


def translate_auto_to_english(text: str) -> str:
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text


def translate_urdu_to_english(text: str) -> str:
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="ur", target="en").translate(text)
    except Exception:
        return text


def translate_english_to_urdu(text: str) -> str:
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="en", target="ur").translate(text)
    except Exception:
        return text


def build_query_lang_profile(query: str) -> QueryLangProfile:
    q = (query or "").strip()

    if is_urdu_script(q) or _safe_langdetect(q) == "ur":
        q_en = translate_urdu_to_english(q)
        return QueryLangProfile(original_query=q, input_lang="ur", query_en=q_en, output_lang="ur")

    if looks_like_roman_urdu(q):
        q_en = translate_auto_to_english(q)
        return QueryLangProfile(original_query=q, input_lang="roman_ur", query_en=q_en, output_lang="roman_ur")

    return QueryLangProfile(original_query=q, input_lang="en", query_en=q, output_lang="en")


def analyze_query(query: str) -> Dict[str, Any]:
    q_lower = (query or "").lower().strip()

    procedural_markers = ["how to", "how do i", "how can i", "steps", "step by step", "procedure"]
    roman_markers = ["kaise", "kesy", "kese", "kaisay", "kya", "kyun", "kyu"]

    list_keywords = ["list", "points", "bullet", "enumerate", "ways", "methods"]
    summary_keywords = ["summarize", "summary", "briefly", "overview", "خلاصہ", "khulasa"]
    detail_keywords = ["explain", "detail", "describe", "why", "تفصیل", "tafseel"]

    wants_steps = any(m in q_lower for m in procedural_markers) or any(m in q_lower for m in roman_markers)
    wants_list = wants_steps or any(kw in q_lower for kw in list_keywords)
    wants_summary = any(kw in q_lower for kw in summary_keywords)
    wants_detail = any(kw in q_lower for kw in detail_keywords)

    return {
        "wants_list": wants_list,
        "wants_summary": wants_summary,
        "wants_detail": wants_detail,
        "is_urdu": is_urdu_script(query) or _safe_langdetect(query) == "ur",
    }


def select_retrieval_queries(profile: QueryLangProfile, query_en_expanded: str, dual: bool) -> List[str]:
    """
    Option-3 retrieval:
    - Urdu-script: use Urdu query embedding (cross-lingual), and optionally English translation (dual)
    - Roman Urdu: use English translation for stable retrieval (docs are English)
    - English: use English
    """
    if profile.input_lang == "ur":
        return [profile.original_query, query_en_expanded] if dual else [profile.original_query]
    if profile.input_lang == "roman_ur":
        return [query_en_expanded]
    return [query_en_expanded]


def urdu_output_instructions() -> str:
    brand_terms = ", ".join(BRAND_TERMS) if BRAND_TERMS else ""
    return f"""
IMPORTANT OUTPUT RULES (Urdu Nastaliq only):
- جواب صرف اردو (نستعلیق) میں دیں۔
- انگریزی الفاظ/رومن اردو استعمال نہ کریں۔
- صرف یہ برانڈ/اداروں کے نام انگریزی میں لکھ سکتے ہیں (اگر ضروری ہوں): {brand_terms}
- اگر معلومات کافی نہیں تو واضح طور پر کہیں کہ متعلقہ معلومات دستیاب نہیں۔
""".strip()


def remove_latin_except_brands(text: str) -> str:
    if not text:
        return text

    protected = {}
    for i, term in enumerate(BRAND_TERMS):
        key = f"@@BRAND_{i}@@"
        protected[key] = term
        text = re.sub(rf"\b{re.escape(term)}\b", key, text, flags=re.IGNORECASE)

    text = re.sub(r"[A-Za-z]{2,}", "", text)

    for key, term in protected.items():
        text = text.replace(key, term)

    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text
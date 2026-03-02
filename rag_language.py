# rag_language.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag_config import (
    BRAND_TERMS,
    ROMAN_URDU_ENABLE,
    ROMAN_URDU_MARKERS,
    ROMAN_URDU_MAX_TOKENS,
    ROMAN_URDU_PAK_VERBIAGE,
    ROMAN_URDU_ROMANIZATION_MODE,
    ROMAN_URDU_STRICT_LATIN_ONLY,
)

_URDU_BLOCK_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_ONLY_RE = re.compile(r"^[\x00-\x7F\s]+$")


# -----------------------------
# Types
# -----------------------------
@dataclass
class QueryLangProfile:
    original_query: str
    input_lang: str          # 'en' | 'ur' | 'roman_ur'
    query_en: str            # English form for retrieval / domain classification
    output_lang: str         # 'en' | 'ur' | 'roman_ur'


# -----------------------------
# Script / Language detection
# -----------------------------
def is_urdu_script(text: str) -> bool:
    return bool(_URDU_BLOCK_RE.search(text or ""))


def _safe_langdetect(text: str) -> str:
    try:
        import langdetect
        return langdetect.detect(text)
    except Exception:
        return "en"


def looks_like_roman_urdu(text: str) -> bool:
    """
    Heuristic: ascii + contains roman-urdu markers.
    """
    if not ROMAN_URDU_ENABLE:
        return False
    if is_urdu_script(text):
        return False
    if re.search(r"[^\x00-\x7F]", text or ""):
        return False
    tokens = re.findall(r"[a-zA-Z']+", (text or "").lower())
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in ROMAN_URDU_MARKERS)
    return (hits >= 2) or (hits >= 1 and len(tokens) <= 6)


# -----------------------------
# Brand protection (important for translation + romanization)
# -----------------------------
def protect_brand_terms(text: str) -> str:
    """
    Wrap brand terms so translation/romanization doesn't corrupt them.
    """
    if not text:
        return text
    out = text
    for term in BRAND_TERMS:
        out = re.sub(rf"\b{re.escape(term)}\b", f"@@{term}@@", out, flags=re.IGNORECASE)
    return out


def restore_brand_terms(text: str) -> str:
    return (text or "").replace("@@", "")


# -----------------------------
# Translation (Deep Translator)
# -----------------------------
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


def translate_english_to_urdu(text: str, timeout: int = 10) -> str:
    """
    Same contract as your RAG_supabase.py implementation (timeouted translation)
    :contentReference[oaicite:4]{index=4}
    """
    try:
        from deep_translator import GoogleTranslator
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

        def _translate():
            return GoogleTranslator(source="en", target="ur").translate(text)

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_translate)
            try:
                result = fut.result(timeout=timeout)
                return result if result else text
            except FutureTimeoutError:
                return text
    except Exception:
        return text


# -----------------------------
# Query profile
# -----------------------------
def build_query_lang_profile(query: str) -> QueryLangProfile:
    q = (query or "").strip()

    if is_urdu_script(q) or _safe_langdetect(q) == "ur":
        q_en = translate_urdu_to_english(q)
        return QueryLangProfile(original_query=q, input_lang="ur", query_en=q_en, output_lang="ur")

    if looks_like_roman_urdu(q):
        # Roman Urdu -> English for retrieval (docs are English)
        q_en = translate_auto_to_english(q)
        return QueryLangProfile(original_query=q, input_lang="roman_ur", query_en=q_en, output_lang="roman_ur")

    return QueryLangProfile(original_query=q, input_lang="en", query_en=q, output_lang="en")


# -----------------------------
# Query analysis (format intent)
# -----------------------------
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
    Your current intended behavior:
    - Urdu-script: optionally use both Urdu + English (dual)
    - Roman Urdu: use English translation for stable retrieval (docs are English)
    - English: use English
    :contentReference[oaicite:5]{index=5}
    """
    if profile.input_lang == "ur":
        return [profile.original_query, query_en_expanded] if dual else [profile.original_query]
    if profile.input_lang == "roman_ur":
        return [query_en_expanded]
    return [query_en_expanded]


# -----------------------------
# Output instructions
# -----------------------------
def urdu_output_instructions() -> str:
    brand_terms = ", ".join(BRAND_TERMS) if BRAND_TERMS else ""
    return f"""
IMPORTANT OUTPUT RULES (Urdu Nastaliq only):
- جواب صرف اردو (نستعلیق) میں دیں۔
- انگریزی الفاظ/رومن اردو استعمال نہ کریں۔
- صرف یہ برانڈ/اداروں کے نام انگریزی میں لکھ سکتے ہیں (اگر ضروری ہوں): {brand_terms}
- اگر معلومات کافی نہیں تو واضح طور پر کہیں کہ متعلقہ معلومات دستیاب نہیں۔
""".strip()


def roman_urdu_output_instructions() -> str:
    brand_terms = ", ".join(BRAND_TERMS) if BRAND_TERMS else ""
    pak = "Pakistani" if ROMAN_URDU_PAK_VERBIAGE else "natural"
    return f"""
IMPORTANT OUTPUT RULES (Roman Urdu - Latin script Urdu):
- Jawab sirf Roman Urdu (Latin letters) mein dein. Urdu/Arabic script mat likhein.
- Zyada Hindi-style spellings avoid karein; {pak} verbiage use karein (e.g., 'aap', 'hain', 'karein', 'nahin').
- Brand names is tarah hi rakhein: {brand_terms}
- Phone numbers/URLs bilkul same rakhein.
- Agar info context mein nahi hai to seedha likhein: "Mujhe maloom nahi".
""".strip()


# -----------------------------
# Roman Urdu: LLM romanization + fallback transliteration
# -----------------------------
def _latin_only_ok(text: str) -> bool:
    if not text:
        return False
    if is_urdu_script(text):
        return False
    if ROMAN_URDU_STRICT_LATIN_ONLY and not _LATIN_ONLY_RE.match(text):
        return False
    return True


def _normalize_pak_roman_urdu(text: str) -> str:
    """
    Light normalization to lean “Pakistani” Roman Urdu.
    """
    if not text:
        return text

    t = text.strip()

    # common normalization
    t = re.sub(r"\bnh(i|y)\b", "nahin", t, flags=re.IGNORECASE)
    t = re.sub(r"\bnai\b", "nahin", t, flags=re.IGNORECASE)
    t = re.sub(r"\bkyun\b", "kyun", t, flags=re.IGNORECASE)
    t = re.sub(r"\bkyu\b", "kyun", t, flags=re.IGNORECASE)
    t = re.sub(r"\bkese\b", "kaise", t, flags=re.IGNORECASE)
    t = re.sub(r"\bkesy\b", "kaise", t, flags=re.IGNORECASE)

    # keep "aap" consistent
    t = re.sub(r"\bap\b", "aap", t, flags=re.IGNORECASE)

    # reduce repeated spaces
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    return t


def romanize_to_roman_urdu_with_llm(urdu_text: str, max_tokens: Optional[int] = None) -> str:
    """
    High quality romanization:
      1) Try direct Urdu-script -> Roman Urdu using the LLM
      2) If it fails, translate Urdu->English and ask LLM to write Roman Urdu
      3) If still fails, fall back (caller can do transliteration fallback)

    This is the modularized version of your existing implementation
    :contentReference[oaicite:6]{index=6}
    """
    urdu_text = (urdu_text or "").strip()
    if not urdu_text:
        return urdu_text

    if max_tokens is None:
        max_tokens = ROMAN_URDU_MAX_TOKENS

    if ROMAN_URDU_ROMANIZATION_MODE == "fallback_only":
        return ""

    # local import to avoid cycles
    try:
        from rag_llm import llm_generate
    except Exception:
        llm_generate = None

    if llm_generate is None:
        return ""

    # Try: Urdu -> Roman Urdu
    for _ in range(2):
        prompt = f"""Task: Convert Urdu (Arabic script) to Roman Urdu (Latin letters).

STRICT RULES:
- Output MUST be in Latin letters only (a-z). No Urdu/Arabic characters at all.
- If you output any Urdu/Arabic characters, your answer is INVALID.
- Keep meaning exactly the same.
- Keep proper names like Alkhidmat as "Alkhidmat".
- Keep phone numbers/URLs unchanged.
- Output ONLY the Roman Urdu text (no labels).
- Use Pakistani Roman Urdu spellings: aap, hain, karein, nahin.

Urdu:
{urdu_text}

Roman Urdu (Latin-only):"""

        out, _, _ = llm_generate(
            prompt,
            max_tokens=max_tokens,
            stop_tokens=["\n\n", "\nUrdu:", "\nRoman Urdu:"],
            language="ur",  # uses Urdu LLM if enabled
        )
        out = (out or "").strip()
        out = _normalize_pak_roman_urdu(out)

        if _latin_only_ok(out):
            return out

    # Try: Urdu -> English -> Roman Urdu
    en = translate_urdu_to_english(urdu_text)
    prompt2 = f"""Task: Write the following English in Roman Urdu (Latin letters).

STRICT RULES:
- Output MUST be in Latin letters only (a-z). No Urdu/Arabic characters.
- Keep meaning exactly the same.
- Keep proper names like Alkhidmat as "Alkhidmat".
- Keep phone numbers/URLs unchanged.
- Output ONLY Roman Urdu text (no labels).
- Use Pakistani Roman Urdu spellings: aap, hain, karein, nahin.

English:
{en}

Roman Urdu (Latin-only):"""

    out2, _, _ = llm_generate(prompt2, max_tokens=max_tokens, stop_tokens=["\n\n"], language=None)
    out2 = _normalize_pak_roman_urdu((out2 or "").strip())

    if _latin_only_ok(out2):
        return out2

    return ""


def transliterate_urdu_to_roman_fallback(urdu_text: str) -> str:
    """
    Deterministic fallback transliteration (not perfect, but safe + offline).
    Goal: ensure you ALWAYS return Latin script when romanization fails.
    """
    s = (urdu_text or "").strip()
    if not s:
        return s

    # Protect brands first (keep exact casing/spelling)
    protected = protect_brand_terms(s)

    # Very lightweight Urdu->Latin mapping (character-level)
    # NOTE: Urdu is not purely phonetic; this is “best effort”
    mapping = {
        "ا": "a", "آ": "aa", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s",
        "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ڈ": "d",
        "ذ": "z", "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh", "س": "s",
        "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "ع": "a",
        "غ": "gh", "ف": "f", "ق": "q", "ک": "k", "گ": "g", "ل": "l",
        "م": "m", "ن": "n", "ں": "n", "و": "w", "ہ": "h", "ھ": "h",
        "ء": "", "ی": "y", "ے": "e",

        "۔": ".", "،": ",", "؟": "?", "٬": ",",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    }

    out_chars = []
    for ch in protected:
        out_chars.append(mapping.get(ch, ch))

    out = "".join(out_chars)
    out = restore_brand_terms(out)

    # remove any remaining Urdu chars if strict
    out = re.sub(r"[\u0600-\u06FF]+", "", out)

    out = _normalize_pak_roman_urdu(out)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()

    # If somehow still not Latin-only, fall back to English translation (last resort)
    if ROMAN_URDU_STRICT_LATIN_ONLY and not _LATIN_ONLY_RE.match(out):
        out = translate_urdu_to_english(urdu_text)

    return out


def to_roman_urdu(urdu_text: str) -> str:
    """
    One public helper:
      - Try LLM romanization
      - If it fails, deterministic transliteration fallback
    """
    s = (urdu_text or "").strip()
    if not s:
        return s

    if ROMAN_URDU_ROMANIZATION_MODE == "fallback_only":
        return transliterate_urdu_to_roman_fallback(s)

    llm_out = romanize_to_roman_urdu_with_llm(s, max_tokens=ROMAN_URDU_MAX_TOKENS)
    if llm_out and _latin_only_ok(llm_out):
        return llm_out

    return transliterate_urdu_to_roman_fallback(s)
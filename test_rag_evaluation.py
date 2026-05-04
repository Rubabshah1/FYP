#!/usr/bin/env python3
"""
ALKHIDMAT RAG - DeepEval Evaluation Suite
==========================================
Tests the RAG pipeline against DeepEval metrics.
Supports text-only AND image+text (multimodal) test cases.

HOW TO RUN:
    pip install deepeval

    # English (default)
    python test_rag_evaluation.py --use-openai-judge

    # Urdu
    python test_rag_evaluation.py --language urdu --use-openai-judge

    # Roman Urdu
    python test_rag_evaluation.py --language roman --use-openai-judge

    # IMAGE test cases
    python test_rag_evaluation.py --language images --use-openai-judge

    # Quick smoke test (3 cases only)
    python test_rag_evaluation.py --language english --max-cases 3 --use-openai-judge

    # Skip specific cases
    python test_rag_evaluation.py --skip Q38 Q39 --use-openai-judge

    # Override image folder (default: C:\\Users\\PC\\fyp\\Test_Images)
    python test_rag_evaluation.py --language images --image-dir path/to/images --use-openai-judge

OUTPUT FILES (fixed paths, re-running overwrites previous results):
    evaluation_results/report_english.json
    evaluation_results/report_urdu.json
    evaluation_results/report_roman.json
    evaluation_results/report_images.json

METRICS (text):
    AnswerRelevancy      -> Is the answer relevant to the question?
    Faithfulness         -> Is the answer grounded in retrieved context?
    ContextualPrecision  -> Are retrieved chunks actually useful?
    ContextualRecall     -> Did retrieval cover what was needed?
    Hallucination        -> Does the answer contain hallucinated facts?

METRICS (image — additional):
    ImageAnswerRelevancy -> Does the answer address what the image shows?
    ImageDescriptionQuality -> How well did GPT-4o describe the image for RAG?
    (These are scored via GPT judge, stored in the report alongside standard metrics)
"""

import os
import sys
import json
import time
import base64
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# ── DeepEval imports ─────────────────────────────────────────────────────────
try:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase
    from deepeval.models.base_model import DeepEvalBaseLLM
except ImportError:
    print("DeepEval not installed. Run: pip install deepeval")
    sys.exit(1)

# ── Suppress noisy logs ───────────────────────────────────────────────────────
import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# ── OpenAI (used for image description) ──────────────────────────────────────
try:
    import openai
    from dotenv import load_dotenv
    load_dotenv()
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")
    _OPENAI_AVAILABLE = bool(openai.api_key)
except ImportError:
    _OPENAI_AVAILABLE = False

# ============================================================================
# LANGUAGE / MODE CONFIG
# Maps --language arg → test case file + fixed output report filename
# ============================================================================
DEFAULT_IMAGE_DIR = r"C:\Users\PC\fyp\Test_Images"

LANGUAGE_CONFIG = {
    "english": {
        "test_file":   "Test_cases(English).json",
        "report_name": "report_english.json",
        "label":       "English",
        "is_image":    False,
    },
    "urdu": {
        "test_file":   "Test_cases(Urdu).json",
        "report_name": "report_urdu.json",
        "label":       "Urdu",
        "is_image":    False,
    },
    "roman": {
        "test_file":   "Test_cases(Roman).json",
        "report_name": "report_roman.json",
        "label":       "Roman Urdu",
        "is_image":    False,
    },
    "images": {
        "test_file":   "Test_cases(Images).json",
        "report_name": "report_images.json",
        "label":       "Image+Text",
        "is_image":    True,
    },
}


# ============================================================================
# NUMPY-SAFE JSON ENCODER
# ============================================================================
import numpy as np

class NumpySafeEncoder(json.JSONEncoder):
    """Converts numpy types to native Python types before JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ============================================================================
# SELF-RAG REJECTION PHRASES
# ============================================================================
RAG_REJECTION_PHRASES = [
    "i don't have enough confidence",
    "i cannot provide a reliable answer",
    "i don't have that information",
    "i found some documents but they don't seem relevant",
    "i cannot provide a sufficiently useful answer",
    "that is an irrelevant question",
    "i apologize, but i couldn't find relevant information",
    "i don't know",
]

def is_rag_rejection(answer: str) -> bool:
    lowered = answer.lower().strip()
    return any(phrase in lowered for phrase in RAG_REJECTION_PHRASES)


# ============================================================================
# LOCAL LLM JUDGE (fallback)
# ============================================================================
class LocalLlamaJudge(DeepEvalBaseLLM):
    def __init__(self):
        self._model = None

    def load_model(self):
        if self._model is None:
            print("[JUDGE] Loading local LLM for evaluation judging...")
            print("[JUDGE] WARNING: Llama 3.2 3B may fail JSON output. Use --use-openai-judge.")
            from rag_llm import load_llm
            self._model = load_llm()
        return self._model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        output = model(prompt, max_tokens=512, temperature=0.1, echo=False)
        return output["choices"][0]["text"].strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "Llama-3.2-3B-Instruct (Local)"


# ============================================================================
# IMAGE UTILITIES
# ============================================================================

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Read an image file and return its base64-encoded string.
    Returns None if the file is missing or unreadable.
    """
    path = Path(image_path)
    if not path.exists():
        print(f"   [IMAGE] File not found: {image_path}")
        return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"   [IMAGE] Could not read {image_path}: {e}")
        return None


def get_image_mime_type(image_path: str) -> str:
    """Return the MIME type based on file extension."""
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/png")


def describe_image_with_gpt4v(
    image_path: str,
    question: str,
    model: str = "gpt-4o",
) -> Tuple[str, str]:
    """
    Send the image + question to GPT-4o Vision and get back:
      - image_description : a detailed description of the image content
      - augmented_query   : question rephrased with image context embedded

    This description is then used as the query to your text-only RAG pipeline,
    making it effectively multimodal without modifying RAG_supabase.py at all.

    Returns ("", "") on failure.
    """
    if not _OPENAI_AVAILABLE:
        print("   [IMAGE] OpenAI not available — cannot describe image.")
        return "", question  # fall back to plain text query

    b64 = encode_image_to_base64(image_path)
    if b64 is None:
        return "", question

    mime = get_image_mime_type(image_path)

    system_prompt = (
        "You are an assistant that describes images for a RAG (Retrieval-Augmented Generation) "
        "system for Alkhidmat Foundation Pakistan. "
        "When given an image and a user question, produce TWO outputs separated by '|||':\n"
        "1. A detailed, factual description of all visible text, logos, numbers, program names, "
        "   contact details, and key visual elements in the image.\n"
        "2. A rewritten version of the user's question that incorporates key details seen in the "
        "   image so it can be used as a standalone search query.\n\n"
        "Format: <image description>|||<augmented query>\n"
        "Do NOT include labels like 'Description:' or 'Query:' — just the two parts separated by |||."
    )

    user_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{b64}",
                "detail": "high",
            },
        },
        {
            "type": "text",
            "text": f"User question: {question}",
        },
    ]

    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=600,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        if "|||" in raw:
            parts = raw.split("|||", 1)
            image_description = parts[0].strip()
            augmented_query   = parts[1].strip()
        else:
            # GPT didn't use the separator — treat whole response as description
            image_description = raw
            augmented_query   = f"{question} [Image context: {raw[:200]}]"

        return image_description, augmented_query

    except Exception as e:
        print(f"   [IMAGE] GPT-4o Vision error: {e}")
        return "", question


def score_image_answer_relevancy(
    question: str,
    image_description: str,
    actual_answer: str,
    expected_answer: str,
    model: str = "gpt-4o-mini",
) -> Dict:
    """
    Ask GPT to score how well the RAG answer addresses what was visible in the image.

    Returns a dict with:
        image_answer_relevancy  : float 0.0–1.0
        image_description_quality : float 0.0–1.0
        image_judge_reason : str
    """
    if not _OPENAI_AVAILABLE or not image_description:
        return {
            "image_answer_relevancy":     None,
            "image_description_quality":  None,
            "image_judge_reason":         "Skipped: OpenAI unavailable or no image description",
        }

    prompt = f"""You are evaluating a RAG chatbot for Alkhidmat Foundation Pakistan.

The user uploaded an IMAGE and asked a question. Evaluate the chatbot's response.

USER QUESTION:
{question}

IMAGE CONTENT (extracted by GPT-4o Vision):
{image_description}

CHATBOT ANSWER:
{actual_answer}

EXPECTED ANSWER:
{expected_answer}

Score the following (respond in JSON only, no markdown):
{{
  "image_answer_relevancy": <float 0.0-1.0 — does the answer address what the image shows?>,
  "image_description_quality": <float 0.0-1.0 — how useful/accurate was the image description for retrieval?>,
  "reason": "<one sentence explanation>"
}}"""

    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return {
            "image_answer_relevancy":    round(float(data.get("image_answer_relevancy", 0.0)), 4),
            "image_description_quality": round(float(data.get("image_description_quality", 0.0)), 4),
            "image_judge_reason":        data.get("reason", ""),
        }
    except Exception as e:
        return {
            "image_answer_relevancy":    None,
            "image_description_quality": None,
            "image_judge_reason":        f"Judge error: {e}",
        }


# ============================================================================
# LOAD TEST CASES
# ============================================================================
def load_test_cases(json_path: str) -> List[Dict]:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Test cases file not found: {json_path}\n"
            f"Make sure the file is in the same directory as this script."
        )

    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    required = {"question", "expected_answer"}
    for i, c in enumerate(cases):
        missing = required - set(c.keys())
        if missing:
            raise ValueError(f"Test case {i} is missing fields: {missing}")

    print(f"Loaded {len(cases)} test cases from {path}")
    return cases


# ============================================================================
# CALL RAG PIPELINE  (text query → answer + contexts)
# ============================================================================
def run_rag_query(question: str) -> Tuple[str, List[str], Dict]:
    """
    Calls generate_answer_selfrag with a plain-text query.
    For image test cases the caller passes the GPT-augmented query here,
    so RAG_supabase.py needs zero modifications.
    """
    try:
        from RAG_supabase import generate_answer_selfrag, retrieve_from_supabase
    except ImportError as e:
        raise ImportError(f"Could not import from RAG_supabase.py: {e}")

    result = generate_answer_selfrag(query=question, top_k=5, max_tokens=400)

    answer                = result[0]
    sources               = result[3] if len(result) > 3 else []
    confidence_scores     = result[4] if len(result) > 4 else {}
    domain_classification = result[5] if len(result) > 5 else {}
    selfrag_metrics       = result[6] if len(result) > 6 else {}

    contexts = []
    try:
        retrieved, _, _ = retrieve_from_supabase(question, top_k=5)
        contexts = [r["text"] for r in retrieved if r.get("text")]
    except Exception as e:
        print(f"   Could not re-fetch chunk texts: {e}")
        contexts = [
            f"[Source: {s.get('filename','?')} | similarity: {s.get('similarity',0):.3f}]"
            for s in sources
        ]

    metadata = {
        "confidence_scores":     confidence_scores,
        "domain_classification": domain_classification,
        "selfrag_metrics":       selfrag_metrics,
        "sources":               sources,
    }
    return answer, contexts, metadata


# ============================================================================
# BUILD DEEPEVAL TEST CASE
# ============================================================================
def build_deepeval_test_case(test_case, actual_output, retrieval_context):
    return LLMTestCase(
        input=test_case["question"],
        actual_output=actual_output,
        expected_output=test_case["expected_answer"],
        retrieval_context=retrieval_context,
        context=retrieval_context,
    )


# ============================================================================
# METRICS
# ============================================================================
def build_metrics(judge_model=None):
    kwargs = {"model": judge_model} if judge_model else {}
    return [
        AnswerRelevancyMetric(threshold=0.5, **kwargs),
        FaithfulnessMetric(threshold=0.5, **kwargs),
        ContextualPrecisionMetric(threshold=0.3, **kwargs),
        ContextualRecallMetric(threshold=0.3, **kwargs),
        HallucinationMetric(threshold=0.4, **kwargs),
    ]


# ============================================================================
# SAVE RESULTS  (fixed filename — overwrites previous run for same language)
# ============================================================================
def save_results(results: List[Dict], language: str, output_dir: str = "evaluation_results") -> str:
    Path(output_dir).mkdir(exist_ok=True)

    cfg = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["english"])
    output_path = Path(output_dir) / cfg["report_name"]

    payload = {
        "language":     cfg["label"],
        "language_key": language,
        "run_at":       datetime.now().isoformat(),
        "results":      results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, cls=NumpySafeEncoder)

    print(f"\nResults saved (overwrote previous) to: {output_path}")
    return str(output_path)


# ============================================================================
# PRINT SUMMARY
# ============================================================================
def print_summary(results: List[Dict], is_image_mode: bool = False):
    metric_names = [
        "AnswerRelevancy", "Faithfulness",
        "ContextualPrecision", "ContextualRecall", "Hallucination",
    ]
    image_metric_names = ["ImageAnswerRelevancy", "ImageDescQuality"]

    aggregate  = {m: [] for m in metric_names}
    pass_counts = {m: 0 for m in metric_names}
    rejection_count = sum(1 for r in results if r.get("rag_rejected"))
    error_count     = sum(1 for r in results if r.get("error"))

    img_aggregate = {m: [] for m in image_metric_names}

    for r in results:
        if r.get("rag_rejected") or r.get("error"):
            continue
        for m in metric_names:
            score = r.get("scores", {}).get(m)
            if score is not None:
                aggregate[m].append(score)
                if r.get("passed", {}).get(m):
                    pass_counts[m] += 1
        if is_image_mode:
            img_scores = r.get("image_scores", {})
            for k, mk in [("image_answer_relevancy", "ImageAnswerRelevancy"),
                          ("image_description_quality", "ImageDescQuality")]:
                v = img_scores.get(k)
                if v is not None:
                    img_aggregate[mk].append(v)

    # ── Header ────────────────────────────────────────────────────────────────
    col_w = 122 if is_image_mode else 115
    print("\n" + "=" * col_w)
    print("EVALUATION SUMMARY" + (" [IMAGE MODE]" if is_image_mode else ""))
    print("=" * col_w)

    header = f"{'Q#':<6} {'Question':<45} {'AnsRel':>8} {'Faith':>8} {'CtxPre':>8} {'CtxRec':>8} {'Halluc':>8}"
    if is_image_mode:
        header += f"  {'ImgRel':>8} {'ImgDesc':>8}"
    header += "  Status"
    print(header)
    print("-" * col_w)

    for r in results:
        q_id   = r.get("id", "?")
        q      = r.get("question", "")[:43]
        scores = r.get("scores", {})

        if r.get("error"):            status = "CRASH"
        elif r.get("rag_rejected"):   status = "REJECTED"
        elif r.get("image_error"):    status = "IMG-ERR"
        else: status = "PASS" if r.get("overall_pass") else "FAIL"

        def fmt(key, src=scores):
            v = src.get(key)
            return f"{v:>8.3f}" if v is not None else f"{'N/A':>8}"

        row = f"{q_id:<6} {q:<45}{fmt('AnswerRelevancy')}{fmt('Faithfulness')}{fmt('ContextualPrecision')}{fmt('ContextualRecall')}{fmt('Hallucination')}"
        if is_image_mode:
            img_scores = r.get("image_scores", {})
            row += f"  {fmt('image_answer_relevancy', img_scores)}{fmt('image_description_quality', img_scores)}"
        row += f"  {status}"
        print(row)

    print("-" * col_w)

    total        = len(results)
    scored_total = total - rejection_count - error_count

    # ── Averages row ──────────────────────────────────────────────────────────
    print(f"{'AVG':<6} {'':<45}", end="")
    for m in metric_names:
        vals = aggregate[m]
        avg = sum(vals) / len(vals) if vals else 0.0
        print(f"{avg:>8.3f}", end="")
    if is_image_mode:
        print("  ", end="")
        for mk in image_metric_names:
            vals = img_aggregate[mk]
            avg = sum(vals) / len(vals) if vals else 0.0
            print(f"{avg:>8.3f}", end="")
    print()

    # ── Pass-rate row ─────────────────────────────────────────────────────────
    print(f"{'PASS%':<6} {'':<45}", end="")
    for m in metric_names:
        rate = (pass_counts[m] / scored_total * 100) if scored_total else 0
        print(f"{rate:>7.1f}%", end="")
    if is_image_mode:
        print("  ", end="")
        for mk in image_metric_names:
            # image metrics don't have a binary pass threshold; show avg instead
            vals = img_aggregate[mk]
            avg = sum(vals) / len(vals) if vals else 0.0
            print(f"{avg:>8.3f}", end="")
    print()

    overall_pass = sum(1 for r in results if r.get("overall_pass", False))
    print(f"\n{'='*col_w}")
    print(f"OVERALL PASS   : {overall_pass}/{total}")
    print(f"RAG REJECTIONS : {rejection_count}/{total}")
    print(f"CRASHES        : {error_count}/{total}")
    print(f"{'='*col_w}\n")


# ============================================================================
# IMAGE TEST CASE RUNNER
# ============================================================================
def run_image_test_case(
    tc: Dict,
    image_dir: str,
    vision_model: str = "gpt-4o",
    judge_model_name: str = "gpt-4o-mini",
) -> Dict:
    """
    Full pipeline for a single image test case:
      1. Resolve image path
      2. GPT-4o Vision → image_description + augmented_query
      3. RAG pipeline (text query = augmented_query)
      4. Image-specific scoring (ImageAnswerRelevancy, ImageDescQuality)

    Returns a result dict compatible with the standard text result format,
    with an additional "image_scores" key.
    """
    q_id     = tc.get("id", "?")
    question = tc["question"]
    expected = tc["expected_answer"]
    img_name = tc.get("image_path", "")

    # ── 1. Resolve image path ─────────────────────────────────────────────────
    image_full_path = str(Path(image_dir) / img_name) if img_name else ""
    image_exists    = Path(image_full_path).exists() if image_full_path else False

    if not image_exists:
        print(f"   [IMAGE] ⚠️  Not found: {image_full_path}")

    # ── 2. GPT-4o Vision: describe image + augment query ─────────────────────
    vision_start = time.time()
    image_description = ""
    augmented_query   = question  # fallback: plain question

    if image_exists and _OPENAI_AVAILABLE:
        print(f"   [IMAGE] Describing with {vision_model}...")
        image_description, augmented_query = describe_image_with_gpt4v(
            image_full_path, question, model=vision_model
        )
        vision_time = round(time.time() - vision_start, 2)
        print(f"   [IMAGE] Vision time  : {vision_time}s")
        print(f"   [IMAGE] Description  : {image_description[:120]}{'...' if len(image_description) > 120 else ''}")
        print(f"   [IMAGE] Aug. query   : {augmented_query[:120]}{'...' if len(augmented_query) > 120 else ''}")
    else:
        vision_time = 0.0
        if not _OPENAI_AVAILABLE:
            print("   [IMAGE] OpenAI unavailable — using plain question for RAG")

    # ── 3. RAG pipeline ───────────────────────────────────────────────────────
    rag_start = time.time()
    try:
        actual_answer, contexts, metadata = run_rag_query(augmented_query)
        rag_time = round(time.time() - rag_start, 2)
    except Exception as e:
        print(f"   RAG pipeline crashed: {e}")
        traceback.print_exc()
        return {
            "id": q_id, "question": question, "expected_answer": expected,
            "image_path": img_name, "image_full_path": image_full_path,
            "image_description": image_description, "augmented_query": augmented_query,
            "actual_answer": None,
            "rag_time_seconds": round(time.time() - rag_start, 2),
            "vision_time_seconds": vision_time,
            "rag_rejected": False, "scores": {}, "passed": {},
            "overall_pass": False, "error": str(e),
            "image_scores": {},
            "image_error": False,
        }

    print(f"   RAG time     : {rag_time}s")
    print(f"   Contexts     : {len(contexts)} chunks retrieved")
    print(f"   Answer       : {actual_answer[:120]}{'...' if len(actual_answer) > 120 else ''}")

    # ── 4. Image-specific scores ──────────────────────────────────────────────
    image_scores = score_image_answer_relevancy(
        question=question,
        image_description=image_description,
        actual_answer=actual_answer,
        expected_answer=expected,
        model=judge_model_name,
    )
    print(f"   ImgRel       : {image_scores.get('image_answer_relevancy', 'N/A')}")
    print(f"   ImgDescQual  : {image_scores.get('image_description_quality', 'N/A')}")
    print(f"   ImgJudge     : {image_scores.get('image_judge_reason', '')[:80]}")

    conf_scores   = metadata.get("confidence_scores", {}) or {}
    combined_conf = conf_scores.get("combined_confidence", None)
    if combined_conf is not None:
        combined_conf = float(combined_conf)

    return {
        "id": q_id, "question": question, "expected_answer": expected,
        "image_path": img_name, "image_full_path": image_full_path,
        "image_description": image_description,
        "augmented_query": augmented_query,
        "actual_answer": actual_answer,
        "rag_time_seconds": rag_time,
        "vision_time_seconds": vision_time,
        "contexts_retrieved": len(contexts),
        "rag_rejected": is_rag_rejection(actual_answer),
        "scores": {},       # populated by DeepEval loop below
        "passed": {},
        "reasons": {},
        "overall_pass": False,
        "image_scores": image_scores,
        "image_error": not image_exists,
        "metadata": {
            "combined_confidence": combined_conf,
            "domain": metadata.get("domain_classification", {}).get("domain"),
            "selfrag_support": metadata.get("selfrag_metrics", {}).get("support_level"),
            "evidence_coverage": metadata.get("selfrag_metrics", {}).get("evidence_coverage"),
            "retrieval_retried": metadata.get("selfrag_metrics", {}).get("retrieval_retried"),
        },
        # Store for DeepEval scoring
        "_actual_answer": actual_answer,
        "_contexts": contexts,
    }


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================
def run_evaluation(
    language: str = "english",
    max_cases: Optional[int] = None,
    use_local_judge: bool = True,
    skip_ids: Optional[List[str]] = None,
    image_dir: str = DEFAULT_IMAGE_DIR,
    vision_model: str = "gpt-4o",
):
    cfg = LANGUAGE_CONFIG.get(language)
    if not cfg:
        print(f"Unknown language '{language}'. Choose from: {list(LANGUAGE_CONFIG.keys())}")
        sys.exit(1)

    is_image_mode = cfg["is_image"]

    print("\n" + "=" * 80)
    print(f"ALKHIDMAT RAG - DeepEval Evaluation Suite  [{cfg['label']}]")
    if is_image_mode:
        print(f"IMAGE DIR: {image_dir}")
        if not _OPENAI_AVAILABLE:
            print("⚠️  WARNING: OPENAI_API_KEY not set — image description will be skipped,")
            print("             RAG will run on plain question text only.")
    print("=" * 80)

    all_cases = load_test_cases(cfg["test_file"])

    if skip_ids:
        before = len(all_cases)
        all_cases = [c for c in all_cases if c.get("id") not in skip_ids]
        print(f"Skipped {before - len(all_cases)} case(s): {skip_ids}")

    if max_cases:
        all_cases = all_cases[:max_cases]
        print(f"Limiting to first {max_cases} case(s)")

    print(f"Total cases to evaluate: {len(all_cases)}\n")

    # ── Judge setup ───────────────────────────────────────────────────────────
    judge_model = None
    if use_local_judge:
        judge_model = LocalLlamaJudge()
        print("Judge: Local Llama (WARNING: may fail JSON output — use --use-openai-judge)\n")
    else:
        print("Judge: OpenAI (reads OPENAI_API_KEY from .env)\n")

    metrics = build_metrics(judge_model)
    print(f"Metrics: {[type(m).__name__ for m in metrics]}\n")

    # ── Image-mode judge model name (for GPT scoring call) ───────────────────
    img_judge_model = "gpt-4o-mini" if not use_local_judge else "gpt-4o-mini"

    results = []

    for i, tc in enumerate(all_cases):
        q_id     = tc.get("id", f"Q{i+1}")
        question = tc["question"]
        expected = tc["expected_answer"]

        print(f"\n{'─'*80}")
        print(f"[{i+1}/{len(all_cases)}] {q_id}: {question[:75]}")
        if is_image_mode:
            print(f"   Image: {tc.get('image_path', 'N/A')}")

        # ── IMAGE MODE: run vision + RAG ──────────────────────────────────────
        if is_image_mode:
            result = run_image_test_case(
                tc=tc,
                image_dir=image_dir,
                vision_model=vision_model,
                judge_model_name=img_judge_model,
            )

            if result.get("error"):
                results.append(result)
                continue

            actual_answer = result.pop("_actual_answer", result.get("actual_answer", ""))
            contexts      = result.pop("_contexts", [])

            if is_rag_rejection(actual_answer):
                result["rag_rejected"]      = True
                result["rejection_message"] = actual_answer
                result["note"]              = "Self-RAG safety rejection. Not scored."
                results.append(result)
                continue

        # ── TEXT MODE: standard RAG call ──────────────────────────────────────
        else:
            rag_start = time.time()
            try:
                actual_answer, contexts, metadata = run_rag_query(question)
                rag_time = round(time.time() - rag_start, 2)
            except Exception as e:
                print(f"   RAG pipeline crashed: {e}")
                traceback.print_exc()
                results.append({
                    "id": q_id, "question": question, "expected_answer": expected,
                    "actual_answer": None,
                    "rag_time_seconds": round(time.time() - rag_start, 2),
                    "rag_rejected": False, "scores": {}, "passed": {},
                    "overall_pass": False, "error": str(e),
                })
                continue

            print(f"   RAG time   : {rag_time}s")
            print(f"   Contexts   : {len(contexts)} chunks retrieved")
            print(f"   Answer     : {actual_answer[:120]}{'...' if len(actual_answer) > 120 else ''}")

            if is_rag_rejection(actual_answer):
                print(f"   REJECTED by Self-RAG: '{actual_answer[:80]}'")
                conf_scores   = metadata.get("confidence_scores", {}) or {}
                results.append({
                    "id": q_id, "question": question, "expected_answer": expected,
                    "actual_answer": actual_answer,
                    "rag_time_seconds": rag_time,
                    "rag_rejected": True, "rejection_message": actual_answer,
                    "scores": {}, "passed": {}, "overall_pass": False,
                    "metadata": {
                        "combined_confidence": float(
                            conf_scores.get("combined_confidence", 0) or 0
                        ),
                        "domain": metadata.get("domain_classification", {}).get("domain"),
                    },
                    "note": "Self-RAG safety rejection. Not scored.",
                })
                continue

            # Build result shell for text mode
            result = {
                "id": q_id, "question": question, "expected_answer": expected,
                "actual_answer": actual_answer, "rag_time_seconds": rag_time,
                "contexts_retrieved": len(contexts), "rag_rejected": False,
                "scores": {}, "passed": {}, "reasons": {},
                "overall_pass": False,
            }
            # Attach metadata
            conf_scores   = metadata.get("confidence_scores", {}) or {}
            combined_conf = conf_scores.get("combined_confidence", None)
            result["metadata"] = {
                "combined_confidence":
                    float(combined_conf) if combined_conf is not None else None,
                "domain": metadata.get("domain_classification", {}).get("domain"),
                "selfrag_support": metadata.get("selfrag_metrics", {}).get("support_level"),
                "evidence_coverage": metadata.get("selfrag_metrics", {}).get("evidence_coverage"),
                "retrieval_retried": metadata.get("selfrag_metrics", {}).get("retrieval_retried"),
            }

        # ── DeepEval scoring (shared for both modes) ──────────────────────────
        deval_case = build_deepeval_test_case(
            {"question": question, "expected_answer": expected},
            actual_answer,
            contexts,
        )

        scores:  Dict[str, Optional[float]] = {}
        passed:  Dict[str, bool]            = {}
        reasons: Dict[str, str]             = {}

        for metric in metrics:
            metric_name = type(metric).__name__.replace("Metric", "")
            if not contexts and metric_name in (
                "ContextualPrecision", "ContextualRecall", "Faithfulness", "Hallucination"
            ):
                scores[metric_name]  = None
                passed[metric_name]  = False
                reasons[metric_name] = "Skipped: no contexts retrieved"
                continue
            try:
                metric.measure(deval_case)
                scores[metric_name]  = round(float(metric.score), 4)
                passed[metric_name]  = metric.is_successful()
                reasons[metric_name] = metric.reason or ""
                status = "PASS" if metric.is_successful() else "FAIL"
                print(
                    f"   [{status}] {metric_name:<25} "
                    f"score={metric.score:.4f} | {(metric.reason or '')[:80]}"
                )
            except Exception as e:
                print(f"   ERROR in {metric_name}: {e}")
                scores[metric_name]  = None
                passed[metric_name]  = False
                reasons[metric_name] = f"Error: {str(e)}"

        scored_passed = [v for k, v in passed.items() if scores.get(k) is not None]
        overall_pass  = all(scored_passed) if scored_passed else False

        result["scores"]       = scores
        result["passed"]       = passed
        result["reasons"]      = reasons
        result["overall_pass"] = overall_pass
        result["actual_answer"] = actual_answer

        results.append(result)

    print_summary(results, is_image_mode=is_image_mode)
    report_path = save_results(results, language)
    return results, report_path


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Alkhidmat RAG pipeline using DeepEval"
    )
    parser.add_argument(
        "--language", "-l",
        choices=["english", "urdu", "roman", "images"],
        default="english",
        help=(
            "Which test set to evaluate: english | urdu | roman | images  "
            "(default: english)"
        ),
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Limit number of cases, e.g. --max-cases 5 for a quick smoke test",
    )
    parser.add_argument(
        "--use-openai-judge", action="store_true",
        help="Use OpenAI GPT as judge (requires OPENAI_API_KEY in .env) — RECOMMENDED",
    )
    parser.add_argument(
        "--skip", nargs="*", default=None,
        help="Question IDs to skip, e.g. --skip Q38 Q39",
    )
    parser.add_argument(
        "--image-dir", default=DEFAULT_IMAGE_DIR,
        help=(
            f"Directory containing image files for --language images. "
            f"Default: {DEFAULT_IMAGE_DIR}"
        ),
    )
    parser.add_argument(
        "--vision-model", default="gpt-4o",
        help="OpenAI vision model used to describe images (default: gpt-4o)",
    )

    args = parser.parse_args()

    results, report_path = run_evaluation(
        language=args.language,
        max_cases=args.max_cases,
        use_local_judge=not args.use_openai_judge,
        skip_ids=args.skip,
        image_dir=args.image_dir,
        vision_model=args.vision_model,
    )

    print(f"Done! Report saved to: {report_path}")
    print(f"(Re-running with --language {args.language} will overwrite this file)")
    print()
    print("NEXT STEPS:")
    print("  1. High rejection rate (>30%)?      Lower SELFRAG_MIN_CONFIDENCE in rag_config.py")
    print("  2. Low ContextualRecall?             Knowledge base may be missing content")
    print("  3. Low Faithfulness?                 LLM is drifting from retrieved context")
    print("  4. CtxPrecision/Recall = 0?          Expected answers too specific for sentence matching")
    print("  5. Low ImageAnswerRelevancy?         GPT-4o Vision description is weak — try gpt-4o (not mini)")
    print("  6. Low ImageDescQuality?             Image is low-res or text is hard to extract; consider pre-OCR")
    print()
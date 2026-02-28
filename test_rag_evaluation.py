#!/usr/bin/env python3
"""
ALKHIDMAT RAG - DeepEval Evaluation Suite
==========================================
Tests the RAG pipeline against DeepEval metrics.

HOW TO RUN:
    pip install deepeval

    # English (default)
    python test_rag_evaluation.py --use-openai-judge

    # Urdu
    python test_rag_evaluation.py --language urdu --use-openai-judge

    # Roman Urdu
    python test_rag_evaluation.py --language roman --use-openai-judge

    # Quick smoke test (3 cases only)
    python test_rag_evaluation.py --language english --max-cases 3 --use-openai-judge

    # Skip specific cases
    python test_rag_evaluation.py --skip Q38 Q39 --use-openai-judge

OUTPUT FILES (fixed paths, re-running overwrites previous results):
    evaluation_results/report_english.json
    evaluation_results/report_urdu.json
    evaluation_results/report_roman.json

METRICS:
    AnswerRelevancy      -> Is the answer relevant to the question?
    Faithfulness         -> Is the answer grounded in retrieved context?
    ContextualPrecision  -> Are retrieved chunks actually useful?
    ContextualRecall     -> Did retrieval cover what was needed?
    Hallucination        -> Does the answer contain hallucinated facts?
"""

import os
import sys
import json
import time
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

# ============================================================================
# LANGUAGE CONFIG
# Maps --language arg → test case file + fixed output report filename
# ============================================================================
LANGUAGE_CONFIG = {
    "english": {
        "test_file":   "Test_cases(English).json",
        "report_name": "report_english.json",
        "label":       "English",
    },
    "urdu": {
        "test_file":   "Test_cases(Urdu).json",
        "report_name": "report_urdu.json",
        "label":       "Urdu",
    },
    "roman": {
        "test_file":   "Test_cases(Roman).json",
        "report_name": "report_roman.json",
        "label":       "Roman Urdu",
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
# CALL RAG PIPELINE
# ============================================================================
def run_rag_query(question: str) -> Tuple[str, List[str], Dict]:
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
        "confidence_scores": confidence_scores,
        "domain_classification": domain_classification,
        "selfrag_metrics": selfrag_metrics,
        "sources": sources,
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

    # Also store language + run timestamp inside the file itself
    payload = {
        "language":   cfg["label"],
        "language_key": language,
        "run_at":     datetime.now().isoformat(),
        "results":    results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, cls=NumpySafeEncoder)

    print(f"\nResults saved (overwrote previous) to: {output_path}")
    return str(output_path)


# ============================================================================
# PRINT SUMMARY
# ============================================================================
def print_summary(results: List[Dict]):
    metric_names = ["AnswerRelevancy","Faithfulness","ContextualPrecision","ContextualRecall","Hallucination"]
    aggregate = {m: [] for m in metric_names}
    pass_counts = {m: 0 for m in metric_names}
    rejection_count = sum(1 for r in results if r.get("rag_rejected"))
    error_count     = sum(1 for r in results if r.get("error"))

    for r in results:
        if r.get("rag_rejected") or r.get("error"):
            continue
        for m in metric_names:
            score = r.get("scores", {}).get(m)
            if score is not None:
                aggregate[m].append(score)
                if r.get("passed", {}).get(m):
                    pass_counts[m] += 1

    print("\n" + "=" * 115)
    print("EVALUATION SUMMARY")
    print("=" * 115)
    print(f"{'Q#':<6} {'Question':<48} {'AnsRel':>8} {'Faith':>8} {'CtxPre':>8} {'CtxRec':>8} {'Halluc':>8}  Status")
    print("-" * 115)

    for r in results:
        q_id   = r.get("id", "?")
        q      = r.get("question", "")[:46]
        scores = r.get("scores", {})

        if r.get("error"):       status = "CRASH"
        elif r.get("rag_rejected"): status = "REJECTED"
        else: status = "PASS" if r.get("overall_pass") else "FAIL"

        def fmt(key):
            v = scores.get(key)
            return f"{v:>8.3f}" if v is not None else f"{'N/A':>8}"

        print(f"{q_id:<6} {q:<48}{fmt('AnswerRelevancy')}{fmt('Faithfulness')}{fmt('ContextualPrecision')}{fmt('ContextualRecall')}{fmt('Hallucination')}  {status}")

    print("-" * 115)
    total        = len(results)
    scored_total = total - rejection_count - error_count

    print(f"{'AVG':<6} {'':<48}", end="")
    for m in metric_names:
        vals = aggregate[m]
        avg = sum(vals)/len(vals) if vals else 0.0
        print(f"{avg:>8.3f}", end="")
    print()

    print(f"{'PASS%':<6} {'':<48}", end="")
    for m in metric_names:
        rate = (pass_counts[m] / scored_total * 100) if scored_total else 0
        print(f"{rate:>7.1f}%", end="")
    print()

    overall_pass = sum(1 for r in results if r.get("overall_pass", False))
    print(f"\n{'='*115}")
    print(f"OVERALL PASS   : {overall_pass}/{total}")
    print(f"RAG REJECTIONS : {rejection_count}/{total}")
    print(f"CRASHES        : {error_count}/{total}")
    print(f"{'='*115}\n")


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================
def run_evaluation(
    language: str = "english",
    max_cases: Optional[int] = None,
    use_local_judge: bool = True,
    skip_ids: Optional[List[str]] = None,
):
    cfg = LANGUAGE_CONFIG.get(language)
    if not cfg:
        print(f"Unknown language '{language}'. Choose from: {list(LANGUAGE_CONFIG.keys())}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f"ALKHIDMAT RAG - DeepEval Evaluation Suite  [{cfg['label']}]")
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

    judge_model = None
    if use_local_judge:
        judge_model = LocalLlamaJudge()
        print("Judge: Local Llama (WARNING: may fail JSON output — use --use-openai-judge)\n")
    else:
        print("Judge: OpenAI (reads OPENAI_API_KEY from .env)\n")

    metrics = build_metrics(judge_model)
    print(f"Metrics: {[type(m).__name__ for m in metrics]}\n")

    results = []

    for i, tc in enumerate(all_cases):
        q_id     = tc.get("id", f"Q{i+1}")
        question = tc["question"]
        expected = tc["expected_answer"]

        print(f"\n{'─'*80}")
        print(f"[{i+1}/{len(all_cases)}] {q_id}: {question[:75]}")

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

        rejected = is_rag_rejection(actual_answer)
        if rejected:
            print(f"   REJECTED by Self-RAG: '{actual_answer[:80]}'")
            results.append({
                "id": q_id, "question": question, "expected_answer": expected,
                "actual_answer": actual_answer, "rag_time_seconds": rag_time,
                "rag_rejected": True, "rejection_message": actual_answer,
                "scores": {}, "passed": {}, "overall_pass": False,
                "metadata": {
                    "combined_confidence": float(metadata.get("confidence_scores", {}).get("combined_confidence", 0) or 0),
                    "domain": metadata.get("domain_classification", {}).get("domain"),
                },
                "note": "Self-RAG safety rejection. Not scored.",
            })
            continue

        deval_case = build_deepeval_test_case(tc, actual_answer, contexts)

        scores: Dict[str, Optional[float]] = {}
        passed: Dict[str, bool]            = {}
        reasons: Dict[str, str]            = {}

        for metric in metrics:
            metric_name = type(metric).__name__.replace("Metric", "")
            if not contexts and metric_name in ("ContextualPrecision","ContextualRecall","Faithfulness","Hallucination"):
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
                print(f"   [{status}] {metric_name:<25} score={metric.score:.4f} | {(metric.reason or '')[:80]}")
            except Exception as e:
                print(f"   ERROR in {metric_name}: {e}")
                scores[metric_name]  = None
                passed[metric_name]  = False
                reasons[metric_name] = f"Error: {str(e)}"

        scored_passed = [v for k, v in passed.items() if scores.get(k) is not None]
        overall_pass  = all(scored_passed) if scored_passed else False

        conf_scores   = metadata.get("confidence_scores", {}) or {}
        combined_conf = conf_scores.get("combined_confidence", None)
        if combined_conf is not None:
            combined_conf = float(combined_conf)

        results.append({
            "id": q_id, "question": question, "expected_answer": expected,
            "actual_answer": actual_answer, "rag_time_seconds": rag_time,
            "contexts_retrieved": len(contexts), "rag_rejected": False,
            "scores": scores, "passed": passed, "reasons": reasons,
            "overall_pass": overall_pass,
            "metadata": {
                "combined_confidence": combined_conf,
                "domain": metadata.get("domain_classification", {}).get("domain"),
                "selfrag_support": metadata.get("selfrag_metrics", {}).get("support_level"),
                "evidence_coverage": metadata.get("selfrag_metrics", {}).get("evidence_coverage"),
                "retrieval_retried": metadata.get("selfrag_metrics", {}).get("retrieval_retried"),
            },
        })

    print_summary(results)
    report_path = save_results(results, language)
    return results, report_path


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Alkhidmat RAG pipeline using DeepEval")
    parser.add_argument(
        "--language", "-l",
        choices=["english", "urdu", "roman"],
        default="english",
        help="Which test set to evaluate: english | urdu | roman  (default: english)",
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

    args = parser.parse_args()

    results, report_path = run_evaluation(
        language=args.language,
        max_cases=args.max_cases,
        use_local_judge=not args.use_openai_judge,
        skip_ids=args.skip,
    )

    print(f"Done! Report saved to: {report_path}")
    print(f"(Re-running with --language {args.language} will overwrite this file)")
    print()
    print("NEXT STEPS:")
    print("  1. High rejection rate (>30%)? Lower SELFRAG_MIN_CONFIDENCE in rag_config.py")
    print("  2. Low ContextualRecall?        Knowledge base may be missing content")
    print("  3. Low Faithfulness?            LLM is drifting from retrieved context")
    print("  4. CtxPrecision/Recall = 0?     Expected answers too specific for sentence matching")
    print()
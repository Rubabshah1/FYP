import gc
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
from llama_cpp import Llama

from rag_config import (
    LLM_MODEL_FILENAME,
    URDU_LLM_ENABLE,
    URDU_LLM_MODEL_FILENAME,
    URDU_LLM_LOAD_VIA_HF,
    URDU_LLM_HF_REPO,
    URDU_LLM_HF_FILENAME,
)

# Multi-model cache (keyed by a stable string)
_LLM_MODELS: Dict[str, Llama] = {}


def detect_apple_silicon() -> bool:
    import platform
    import subprocess

    try:
        if platform.system() != "Darwin":
            return False
        if platform.machine() == "arm64":
            return True
        result = subprocess.run(["uname", "-m"], capture_output=True, text=True)
        return (result.returncode == 0 and "arm64" in (result.stdout or ""))
    except Exception:
        return False


def _make_llama_kwargs() -> dict:
    is_apple = detect_apple_silicon()
    gpu_layers = -1 if is_apple else 0
    return dict(
        n_ctx=4096,
        n_gpu_layers=gpu_layers,
        verbose=False,
        logits_all=True,
    )


def load_llm_local(model_filename: str) -> Llama:
    cache_key = f"local::{model_filename}"
    if cache_key in _LLM_MODELS:
        return _LLM_MODELS[cache_key]

    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file missing: {model_filename}")

    print("Loading local LLM via llama-cpp:", model_filename)
    model = Llama(model_path=model_filename, **_make_llama_kwargs())
    _LLM_MODELS[cache_key] = model
    return model


def load_llm_urdu() -> Llama:
    """
    Urdu model loading strategy:
    - If URDU_LLM_LOAD_VIA_HF=1: download GGUF from Hugging Face using Llama.from_pretrained()
    - else: load local GGUF file path URDU_LLM_MODEL_FILENAME
    """
    if not URDU_LLM_ENABLE:
        return load_llm_local(LLM_MODEL_FILENAME)

    if URDU_LLM_LOAD_VIA_HF:
        cache_key = f"hf::{URDU_LLM_HF_REPO}::{URDU_LLM_HF_FILENAME}"
        if cache_key in _LLM_MODELS:
            return _LLM_MODELS[cache_key]

        print("Loading Urdu LLM from Hugging Face (GGUF):", URDU_LLM_HF_REPO, URDU_LLM_HF_FILENAME)
        model = Llama.from_pretrained(
            repo_id=URDU_LLM_HF_REPO,
            filename=URDU_LLM_HF_FILENAME,
            **_make_llama_kwargs(),
        )
        _LLM_MODELS[cache_key] = model
        return model

    return load_llm_local(URDU_LLM_MODEL_FILENAME)


def load_llm_default() -> Llama:
    return load_llm_local(LLM_MODEL_FILENAME)


def get_llm_for_language(language: Optional[str]) -> Llama:
    if language == "ur" and URDU_LLM_ENABLE:
        print("[DEBUG] Selecting Urdu-specific local model")
        return load_llm_urdu()
    print("[DEBUG] Selecting default local model")
    return load_llm_default()


# -----------------------------
# OpenAI (optional)
# -----------------------------
_OPENAI_CLIENT = None


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OPENAI_API_KEY is set but the 'openai' package is not installed. Run: pip install openai"
        ) from e

    _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def _apply_stop_tokens(text: str, stop_tokens: list | None) -> str:
    if not stop_tokens:
        return text.strip()
    cut_at = None
    for s in stop_tokens:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            cut_at = idx if cut_at is None else min(cut_at, idx)
    if cut_at is not None:
        text = text[:cut_at]
    return text.strip()


def llm_generate(
    prompt: str,
    max_tokens: int = 400,
    stop_tokens: list | None = None,
    language: str | None = None,
) -> Tuple[str, List[float], List[np.ndarray]]:
    """
    Returns: (text, token_logprobs, token_topk_prob_distributions)
    - If OPENAI_API_KEY is set: uses OpenAI with logprobs (if supported)
    - Else: local llama-cpp
    """
    gc.collect()

    client = _get_openai_client()
    if client is not None:
        model_name = os.environ.get("OPENAI_MODEL_URDU" if language == "ur" else "OPENAI_MODEL", "gpt-5.2")
        reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT", "none")
        top_k = int(os.environ.get("OPENAI_TOP_LOGPROBS", "5"))

        try:
            print(f"[DEBUG] Using OpenAI GPT model: {model_name}")
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                logprobs=True,
                top_logprobs=top_k,
            )

            text = (completion.choices[0].message.content or "").strip()
            text = _apply_stop_tokens(text, stop_tokens)

            log_probs: List[float] = []
            token_probs_distributions: List[np.ndarray] = []

            lp = completion.choices[0].logprobs
            if lp and lp.content:
                for tok in lp.content:
                    if tok.logprob is not None:
                        log_probs.append(float(tok.logprob))
                    if tok.top_logprobs:
                        cand_logps = [float(c.logprob) for c in tok.top_logprobs if c.logprob is not None]
                        if cand_logps:
                            probs = np.exp(cand_logps)
                            s = float(np.sum(probs))
                            if s > 0:
                                probs = probs / s
                            token_probs_distributions.append(probs)

            gc.collect()
            return text, log_probs, token_probs_distributions

        except Exception as e:
            print(f"[OpenAI] Generation Error (falling back to local LLM): {e}")

    # local llama-cpp
    try:
        model = get_llm_for_language(language)
        print(f"[DEBUG] Generating response with local Llama model (Language: {language})")
        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.2,
            stop=stop_tokens or [],
            echo=False,
            logprobs=5,
        )

        text = (output["choices"][0]["text"] or "").strip()

        log_probs: List[float] = []
        token_probs_distributions: List[np.ndarray] = []

        lp = output["choices"][0].get("logprobs")
        if lp:
            tlp = lp.get("token_logprobs") or []
            log_probs = [float(x) for x in tlp if x is not None]

            top_lp = lp.get("top_logprobs") or []
            for token_dict in top_lp:
                if token_dict:
                    logprobs_list = list(token_dict.values())
                    probs = np.exp(logprobs_list)
                    denom = float(np.sum(probs))
                    if denom > 0:
                        probs = probs / denom
                    token_probs_distributions.append(probs)

        gc.collect()
        return text, log_probs, token_probs_distributions

    except Exception as e:
        print(f"LLM Generation Error: {e}")
        gc.collect()
        return "Error generating response.", [], []
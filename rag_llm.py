import gc
import os
from typing import List, Tuple

import numpy as np
from llama_cpp import Llama

from rag_config import LLM_MODEL_FILENAME

# -----------------------------
# Local Llama-cpp (existing)
# -----------------------------
_LLM_MODEL: Llama | None = None


def detect_apple_silicon() -> bool:
    """Detect if running on Apple Silicon (M1/M2/M3/etc)."""
    import platform
    import subprocess

    try:
        if platform.system() != "Darwin":
            return False

        machine = platform.machine()
        if machine == "arm64":
            return True

        result = subprocess.run(["uname", "-m"], capture_output=True, text=True)
        if result.returncode == 0 and "arm64" in result.stdout:
            return True

        return False
    except Exception:
        return False


def load_llm(model_filename: str = LLM_MODEL_FILENAME) -> Llama:
    """Load (or return cached) local Llama model via llama-cpp."""
    global _LLM_MODEL
    if _LLM_MODEL is None:
        print("Loading local LLM via llama-cpp:", model_filename)

        if not os.path.exists(model_filename):
            print(f"❌ Error: Model file not found at {model_filename}")
            raise FileNotFoundError(f"Model file missing: {model_filename}")

        is_apple_silicon = detect_apple_silicon()
        gpu_layers = -1 if is_apple_silicon else 0

        if is_apple_silicon:
            print("🍎 Apple Silicon detected - enabling Metal GPU acceleration")
        else:
            print("💻 Using CPU mode (no GPU acceleration)")

        _LLM_MODEL = Llama(
            model_path=model_filename,
            n_ctx=4096,
            n_gpu_layers=gpu_layers,
            verbose=False,
            logits_all=True,  # used by your confidence scorer for token probs
        )

        if is_apple_silicon:
            print("✅ LLM loaded with Metal GPU acceleration")
        else:
            print("✅ LLM loaded in CPU mode")

    return _LLM_MODEL


# -----------------------------
# OpenAI (generation + logprobs)
# -----------------------------
_OPENAI_CLIENT = None


def _get_openai_client():
    """
    Lazy-load OpenAI client.
    Requires: pip install openai
    Env: OPENAI_API_KEY
    """
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
            "OPENAI_API_KEY is set but the 'openai' package is not installed. "
            "Run: pip install openai"
        ) from e

    _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def _apply_stop_tokens(text: str, stop_tokens: list | None) -> str:
    """Client-side stop-token truncation (use this instead of OpenAI stop param)."""
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
    prompt: str, max_tokens: int = 400, stop_tokens: list | None = None
) -> Tuple[str, List[float], List[np.ndarray]]:
    """
    Returns:
      (text, token_logprobs, token_topk_prob_distributions)

    Behavior:
    - If OPENAI_API_KEY is set -> uses OpenAI with logprobs (for Self-RAG)
    - If OpenAI fails -> falls back to local llama-cpp (keeps your system running)
    """
    gc.collect()

    # ---------- 1) OpenAI path ----------
    client = _get_openai_client()
    if client is not None:
        model_name = os.environ.get("OPENAI_MODEL", "gpt-5.2")
        reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT", "none")
        top_k = int(os.environ.get("OPENAI_TOP_LOGPROBS", "5"))

        print("[OpenAI] Using model:", model_name)

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,  # GPT-5 expects this, not max_tokens
                reasoning_effort=reasoning_effort,
                logprobs=True,
                top_logprobs=top_k,
                # Do NOT pass stop; your logs show it's rejected for your model setup.
            )

            text = (completion.choices[0].message.content or "").strip()
            text = _apply_stop_tokens(text, stop_tokens)

            log_probs: List[float] = []
            token_probs_distributions: List[np.ndarray] = []

            lp = completion.choices[0].logprobs
            if lp and lp.content:
                for tok in lp.content:
                    # per-token logprob
                    if tok.logprob is not None:
                        log_probs.append(float(tok.logprob))

                    # distribution over top_k tokens (normalize)
                    if tok.top_logprobs:
                        cand_logps = [
                            float(c.logprob)
                            for c in tok.top_logprobs
                            if c.logprob is not None
                        ]
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

    # ---------- 2) Local fallback ----------
    try:
        model = load_llm()
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

        text = output["choices"][0]["text"].strip()

        log_probs: List[float] = []
        token_probs_distributions: List[np.ndarray] = []

        if "logprobs" in output["choices"][0] and output["choices"][0]["logprobs"]:
            logprobs_data = output["choices"][0]["logprobs"]

            if "token_logprobs" in logprobs_data and logprobs_data["token_logprobs"]:
                log_probs = [lp for lp in logprobs_data["token_logprobs"] if lp is not None]

            if "top_logprobs" in logprobs_data and logprobs_data["top_logprobs"]:
                for token_dict in logprobs_data["top_logprobs"]:
                    if token_dict:
                        logprobs_list = list(token_dict.values())
                        probs = np.exp(logprobs_list)
                        probs = probs / np.sum(probs)
                        token_probs_distributions.append(probs)

        gc.collect()
        return text, log_probs, token_probs_distributions

    except Exception as e:
        print(f"LLM Generation Error: {e}")
        gc.collect()
        return "Error generating response.", [], []
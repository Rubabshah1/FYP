import gc
import os
import time
from typing import List, Tuple

import numpy as np
from llama_cpp import Llama

from rag_config import LLM_MODEL_FILENAME

_LLM_MODEL: Llama | None = None


def detect_apple_silicon() -> bool:
    """Detect if running on Apple Silicon (M1/M2/M3/etc)."""
    import platform
    import subprocess

    try:
        if platform.system() != "Darwin":
            return False

        # Check processor architecture
        machine = platform.machine()
        if machine == "arm64":
            return True

        # Alternative check using uname
        result = subprocess.run(
            ["uname", "-m"], capture_output=True, text=True
        )
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

        # Detect Apple Silicon for Metal GPU acceleration
        is_apple_silicon = detect_apple_silicon()
        gpu_layers = -1 if is_apple_silicon else 0  # -1 = use all GPU layers

        if is_apple_silicon:
            print("🍎 Apple Silicon detected - enabling Metal GPU acceleration")
        else:
            print("💻 Using CPU mode (no GPU acceleration)")

        _LLM_MODEL = Llama(
            model_path=model_filename,
            n_ctx=4096,
            n_gpu_layers=gpu_layers,  # -1 for Apple Silicon (Metal), 0 for CPU
            verbose=False,
            logits_all=True,  # Enable logits for log probability extraction
        )

        if is_apple_silicon:
            print("✅ LLM loaded with Metal GPU acceleration")
        else:
            print("✅ LLM loaded in CPU mode")
    return _LLM_MODEL


def llm_generate(
    prompt: str, max_tokens: int = 400, stop_tokens: list | None = None
) -> Tuple[str, List[float], List[np.ndarray]]:
    """
    MEMORY-OPTIMIZED: Generation with garbage collection.
    Mirrors the original implementation from `RAG_supabase.py`.
    """
    model = load_llm()

    # Force garbage collection before generation
    gc.collect()

    try:
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

        # Extract log probabilities if available
        log_probs: List[float] = []
        token_probs_distributions: List[np.ndarray] = []

        if "logprobs" in output["choices"][0] and output["choices"][0]["logprobs"]:
            logprobs_data = output["choices"][0]["logprobs"]

            if (
                "token_logprobs" in logprobs_data
                and logprobs_data["token_logprobs"]
            ):
                log_probs = [
                    lp for lp in logprobs_data["token_logprobs"] if lp is not None
                ]

            if "top_logprobs" in logprobs_data and logprobs_data["top_logprobs"]:
                for token_dict in logprobs_data["top_logprobs"]:
                    if token_dict:
                        logprobs_list = list(token_dict.values())
                        probs = np.exp(logprobs_list)
                        probs = probs / np.sum(probs)
                        token_probs_distributions.append(probs)

        # Cleanup after generation
        gc.collect()

        return text, log_probs, token_probs_distributions

    except Exception as e:
        print(f"LLM Generation Error: {e}")
        gc.collect()  # Cleanup even on error
        return "Error generating response.", [], []


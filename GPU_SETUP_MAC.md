# GPU Acceleration Setup for Mac (Apple Silicon)

This guide explains how to enable GPU acceleration for the LLM on Apple Silicon Macs (M1, M2, M3, etc.) using Metal.

## Prerequisites

1. **Apple Silicon Mac** (M1, M2, M3, or newer)
   - Check: `uname -m` should return `arm64`
   - Intel Macs don't support Metal GPU acceleration for llama-cpp-python

2. **Metal-enabled llama-cpp-python**
   - The standard pip install may not include Metal support
   - You need to install with Metal support

## Installation Steps

### Option 1: Install llama-cpp-python with Metal Support (Recommended)

**If using a virtual environment (recommended):**
```bash
# Activate your virtual environment first
source venv/bin/activate  # or your venv path

# Uninstall existing version if installed
pip uninstall llama-cpp-python -y

# Install with Metal support (pre-built wheel - fastest)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# OR build with Metal support (if pre-built doesn't work)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir
```

**If using system Python:**
```bash
# Use pip3 with --user flag or create a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### Option 2: Build from Source with Metal

```bash
# Install build dependencies
brew install cmake

# Uninstall existing version
pip uninstall llama-cpp-python -y

# Install with Metal enabled
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
```

### Verify Installation

```bash
python3 -c "from llama_cpp import Llama; print('✅ llama-cpp-python installed')"
```

If you see Metal-related messages when loading the model (like `ggml_metal_init`), Metal support is enabled.

## Configuration

The code automatically detects Apple Silicon and enables GPU acceleration. The `load_llm()` function in `RAG_supabase.py` will:

- **Apple Silicon (arm64):** Set `n_gpu_layers=-1` (offload all layers to GPU)
- **Intel Mac:** Set `n_gpu_layers=0` (CPU only)
- **Other systems:** Set `n_gpu_layers=0` (CPU only)

### Manual Configuration

If you want to control GPU layer offloading manually, edit `RAG_supabase.py`:

```python
# In load_llm() function, change:
gpu_layers = -1  # -1 = all layers on GPU, 0 = CPU only, or specific number like 35
```

**Recommendations:**
- **-1 (all layers):** Best performance, uses more VRAM
- **35-40 layers:** Good balance for 3B models
- **0 (CPU only):** If you have memory issues

## Performance Expectations

With Metal GPU acceleration on Apple Silicon:

| Model Size | CPU Time | GPU Time (Metal) | Speedup |
|-----------|----------|-----------------|---------|
| 3B (Q4_K_M) | ~8-12s | ~3-5s | 2-3x faster |
| 7B (Q4_K_M) | ~20-30s | ~8-12s | 2-3x faster |

**Note:** Actual performance depends on:
- Model quantization (Q4_K_M is faster than Q8_0)
- Context length
- Number of tokens generated
- System load

## Troubleshooting

### Issue: "Metal not available" or still using CPU

**Solution:**
1. Verify Metal support:
   ```bash
   python3 -c "from llama_cpp import Llama; import sys; sys.path.insert(0, '.'); from RAG_supabase import load_llm; load_llm()"
   ```
   Look for `ggml_metal_init` messages in output.

2. Reinstall with Metal:
   ```bash
   pip uninstall llama-cpp-python -y
   CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir
   ```

### Issue: Out of Memory Errors

**Solution:**
- Reduce `n_gpu_layers` to a specific number (e.g., 35 instead of -1)
- Use a smaller quantized model (Q4_K_M instead of Q8_0)
- Reduce `n_ctx` (context window size)

### Issue: Model loads but is slow

**Check:**
1. Verify GPU is being used - look for Metal initialization messages
2. Check Activity Monitor - GPU should show activity
3. Ensure you're using a quantized model (Q4_K_M or similar)

## Testing GPU Acceleration

Run this test to verify GPU is working:

```python
from RAG_supabase import load_llm
import time

print("Loading model...")
model = load_llm()

print("Generating test response...")
start = time.time()
response = model("What is AI?", max_tokens=50)
elapsed = time.time() - start

print(f"Time: {elapsed:.2f}s")
print(f"Response: {response['choices'][0]['text']}")
```

If you see Metal initialization messages and faster generation times, GPU acceleration is working.

## Current Configuration

The code is configured to:
- ✅ Automatically detect Apple Silicon
- ✅ Enable Metal GPU acceleration (`n_gpu_layers=-1`)
- ✅ Fall back to CPU on Intel Macs
- ✅ Print GPU status on model load

## Additional Resources

- [llama-cpp-python Metal Support](https://github.com/abetlen/llama-cpp-python#metal-mac)
- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [llama.cpp Metal Backend](https://github.com/ggerganov/llama.cpp#metal-build)

---

*Last updated: 2024*


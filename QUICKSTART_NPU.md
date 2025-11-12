# Quick Start Guide - NPU Acceleration

## Problem: Ollama doesn't use NPU directly

The original `main.py` uses Ollama's API, which runs separately and uses CPU/RAM.

## Solution: Use ONNX model with NPU directly

### Quick Test (GPT-2 Example)

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   pip install optimum[onnxruntime] transformers
   ```

2. **Export GPT-2 to ONNX:**

   ```bash
   optimum-cli export onnx --model gpt2 ./models/gpt2_onnx
   ```

3. **Run with NPU:**

   ```bash
   python src/main_onnx.py ./models/gpt2_onnx/model.onnx
   ```

4. **Check output for:**
   ```
   ✅✅✅ NPU IS ACTIVE - WILL NOT USE RAM/CPU/GPU HEAVILY ✅✅✅
   ```

### Convert Your Own Model

For DeepSeek or other models:

1. **If model is on Hugging Face:**

   ```bash
   optimum-cli export onnx --model deepseek-ai/deepseek-llm-7b-base ./models/deepseek_onnx
   python src/main_onnx.py ./models/deepseek_onnx/model.onnx
   ```

2. **If model is in Ollama:**
   - You need to extract and convert it (more complex)
   - See `model_converter.py` for guidance

### Verify NPU Usage

**In Task Manager (Windows):**

- CPU usage: Should be LOW (< 10%)
- RAM usage: Should be LOW
- NPU: Should show activity

**In Terminal Output:**

```
Provider: QNNExecutionProvider  ✅ (means NPU active)
Provider: CPUExecutionProvider  ❌ (means using CPU)
```

### Troubleshooting

**"No NPU provider found"**

- Install QNN-enabled onnxruntime
- Check if Qualcomm drivers are installed
- Try DirectML provider: `pip install onnxruntime-directml`

**"Model file not found"**

- Make sure you completed the export step
- Check the file path is correct

**Still using CPU/RAM?**

- The model might be too large for NPU
- Try smaller models first (GPT-2, DistilBERT)
- Check if NPU drivers are properly installed

## Why This Works

- **Ollama**: Runs its own process → Uses CPU/RAM
- **ONNX + NPU**: Loads model directly to NPU → Offloads from CPU/RAM

The `main_onnx.py` bypasses Ollama completely and talks directly to the NPU!

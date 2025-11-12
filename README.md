# Zero Cloud LLM

Local LLM inference with NPU acceleration using ONNX models from Hugging Face.

## 🚀 Features

- ✅ **ONNX Model Support**: Uses optimized ONNX models from Hugging Face
- ⚡ **NPU Acceleration**: Automatic hardware acceleration when available
- 🎯 **DeepSeek-R1**: Pre-configured with DeepSeek-R1-Distill-Qwen-1.5B
- 🌊 **Streaming Responses**: Real-time token streaming
- � **Interactive Mode**: Chat interface in terminal
- � **Clean UI**: Colored output with emoji indicators
- 🔧 **Easy Setup**: Simple npm install and run

## 📋 Prerequisites

- **Node.js 18+** (for ES modules support)
- **~2GB disk space** (for model download on first run)
- **4GB+ RAM** recommended
- **Windows on ARM (Qualcomm)** for best NPU performance

## 🛠️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/krsXishere/zero-cloud-llm.git
   cd zero-cloud-llm
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

   This will install `@huggingface/transformers` which includes:

   - ONNX Runtime with NPU support
   - Tokenizers
   - Model management

## 🎯 Usage

### Interactive Mode (Default)

```bash
npm start
```

or

```bash
node src/main.js
```

This starts an interactive chat session:

- Type your prompts and get real-time responses
- Type `quit` or `exit` to exit
- Type `clear` to clear the screen

### Demo Mode

Run with predefined sample prompts:

```bash
node src/main.js --demo
```

### Single Prompt Mode

Generate a response for a single prompt:

```bash
node src/main.js --prompt "What is quantum computing?"
```

### Quick Test

Test the model with a simple example:

```bash
npm test
```

or

```bash
node src/test.js
```

## 💡 How It Works

### NPU Acceleration

The application uses `@huggingface/transformers` which includes ONNX Runtime with automatic hardware acceleration:

1. **NPU (Neural Processing Unit)**: If available on Windows on ARM (Qualcomm)
2. **GPU**: Falls back to GPU if available
3. **CPU**: Final fallback to CPU

The model `DeepSeek-R1-Distill-Qwen-1.5B-ONNX` is:

- Already in ONNX format (optimized for inference)
- Quantized to `q4f16` (4-bit weights, float16 activations)
- Small size (~1GB) with good performance

### First Run

On the first run, the model will be downloaded automatically:

- **Model**: `onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX`
- **Size**: ~1GB
- **Location**: Cached in `~/.cache/huggingface/`

Subsequent runs will use the cached model (instant startup).

## 🔧 Configuration

Edit `src/main.js` to customize:

```javascript
const CONFIG = {
  model: "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX",
  dtype: "q4f16", // Quantization: q4f16, q8, fp16, fp32
  maxTokens: 512, // Maximum tokens to generate
  temperature: 0.7, // Sampling temperature
};
```

### Available Models

You can use other ONNX models from Hugging Face:

```javascript
// Smaller model (faster)
"onnx-community/Qwen2.5-0.5B-Instruct";

// Larger models (better quality)
"onnx-community/DeepSeek-R1-Distill-Qwen-7B-ONNX";
"onnx-community/Phi-3-mini-4k-instruct-onnx";
```

## 📊 Performance

On Windows on ARM (Qualcomm X Elite):

- **With NPU**: Low CPU/RAM usage, fast inference
- **Without NPU**: Falls back to CPU, still reasonably fast
- **First token**: ~100-500ms (cached model)
- **Streaming**: ~10-50 tokens/second (hardware dependent)

Monitor usage:

- Open Task Manager
- Check CPU/RAM during inference
- With NPU: CPU should stay low (<20%)
- Without NPU: CPU will spike

## 🐛 Troubleshooting

### Model Download Fails

```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/models--onnx-community--DeepSeek-R1-Distill-Qwen-1.5B-ONNX
npm start
```

### Out of Memory

Try a smaller model or reduce max tokens:

```javascript
const CONFIG = {
  model: "onnx-community/Qwen2.5-0.5B-Instruct",
  maxTokens: 256,
};
```

### Slow Performance

1. Check if NPU drivers are installed
2. Try different quantization: `"q8"` or `"fp16"`
3. Reduce `maxTokens`
4. Use a smaller model

## 📁 Project Structure

```
zero-cloud-llm/
├── src/
│   ├── main.js              # Main application (interactive/demo)
│   └── test.js              # Simple test script
├── package.json             # Node.js dependencies
├── README.md               # This file
├── QUICKSTART.md           # Quick start guide
└── .gitignore              # Git ignore rules
```

## 🔄 Comparison: Python vs JavaScript

| Feature       | Python (ONNX)         | JavaScript (Transformers.js) |
| ------------- | --------------------- | ---------------------------- |
| Setup         | Complex               | ✅ Simple (npm install)      |
| NPU Support   | Manual config         | ✅ Automatic                 |
| Model Loading | Manual ONNX export    | ✅ Auto-download from HF     |
| Streaming     | Manual implementation | ✅ Built-in                  |
| Performance   | Similar               | Similar                      |

**Recommendation**: Use JavaScript version for easier setup and better NPU auto-detection.

## 📚 Resources

- [Hugging Face Transformers.js](https://huggingface.co/docs/transformers.js)
- [ONNX Models on Hugging Face](https://huggingface.co/onnx-community)
- [DeepSeek-R1 Model](https://huggingface.co/deepseek-ai)

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📄 License

MIT License

---

Made with ❤️ for Qualcomm NPU acceleration

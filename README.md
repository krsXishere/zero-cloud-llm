# Zero Cloud LLM

Local LLM inference with Qualcomm NPU acceleration on Windows on Snapdragon (ARM64).

## Features

- 🚀 **NPU Acceleration**: Automatically detects and uses Qualcomm NPU when available
- 🔄 **Automatic Fallback**: Falls back to CPU if NPU is not available
- 📡 **Ollama Integration**: Uses local Ollama API for model management
- 🌊 **Streaming Responses**: Real-time token streaming for better UX
- 📊 **Status Logging**: Clear logging of which backend is being used
- 🏗️ **Modular Design**: Clean architecture ready for ONNX model integration

## Prerequisites

1. **Ollama** installed and running locally

   - Download from: https://ollama.ai
   - Default endpoint: `http://localhost:11434`

2. **Python 3.8+** on Windows on Snapdragon

3. **Model pulled in Ollama**:
   ```bash
   ollama pull deepseek-r1:1.5b
   ```

## Installation

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd zero-cloud-llm
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Interactive Mode (Default)

```bash
python src/main.py
```

This starts an interactive chat session where you can:

- Type prompts and get responses
- Type `status` to check NPU acceleration status
- Type `quit` or `exit` to exit

### Demo Mode

Run with predefined prompts:

```bash
python src/main.py --demo
```

### Single Prompt Mode

Generate a response for a single prompt:

```bash
python src/main.py --prompt "What is machine learning?"
```

## Configuration

Edit `src/main.py` to customize:

```python
MODEL_NAME = "deepseek-r1:1.5b"  # Your Ollama model
OLLAMA_URL = "http://localhost:11434"  # Ollama server URL
```

## NPU Acceleration

The application automatically detects the following NPU providers:

- **QNNExecutionProvider** - Qualcomm Neural Network SDK
- **SNPEExecutionProvider** - Snapdragon Neural Processing Engine
- **DMLExecutionProvider** - DirectML (may use NPU on Windows ARM)

If no NPU provider is found, it falls back to CPU.

### Current State

Currently, the application uses Ollama's API which handles inference internally. NPU detection infrastructure is in place for future enhancements:

1. **Phase 1 (Current)**: Ollama API with NPU detection
2. **Phase 2 (Future)**: Direct ONNX model loading with NPU acceleration

To enable Phase 2:

1. Export your model to ONNX format
2. Convert to QNN format if needed
3. Use `onnxruntime` with detected NPU provider
4. See `npu_detector.py` for provider details

## Project Structure

```
zero-cloud-llm/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── main.py               # Main application entry point
│   ├── inference_engine.py   # Main inference orchestration
│   ├── npu_detector.py       # NPU detection and management
│   └── ollama_client.py      # Ollama API client
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Troubleshooting

### Ollama Connection Error

```
✗ Failed to connect to Ollama server
```

**Solution**: Make sure Ollama is running:

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

### Model Not Found

```
Model 'deepseek-r1:1.5b' not found in available models
```

**Solution**: Pull the model first:

```bash
ollama pull deepseek-r1:1.5b
```

### No NPU Provider Found

```
✗ No NPU provider found. Falling back to CPU.
```

This is normal if:

- Running on non-Qualcomm hardware
- NPU drivers not installed
- ONNX Runtime without NPU support

The application will still work using CPU.

## Future Enhancements

- [ ] Direct ONNX model loading with NPU
- [ ] Model conversion utilities (PyTorch/GGUF → ONNX → QNN)
- [ ] Benchmark tools for NPU vs CPU performance
- [ ] Multi-model support
- [ ] Conversation history management
- [ ] Model quantization options

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or PR.

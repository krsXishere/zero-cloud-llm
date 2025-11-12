# Migration to JavaScript

Project ini telah **diubah dari Python ke JavaScript** untuk kemudahan penggunaan NPU.

## ✅ Keuntungan JavaScript Version

1. **Setup Lebih Mudah**: `npm install` vs kompleks ONNX export
2. **Auto NPU Detection**: Otomatis menggunakan NPU jika tersedia
3. **Auto Model Download**: Model ONNX didownload otomatis dari Hugging Face
4. **Built-in Streaming**: Tidak perlu implementasi manual
5. **Instant Start**: Model di-cache, startup cepat

## 🚀 Quick Start

```bash
# Install
npm install

# Run
npm start
```

## 📁 File Python (Legacy)

File Python masih ada di `src/` untuk referensi:

- `npu_detector.py` - NPU detection
- `ollama_client.py` - Ollama integration
- `inference_engine.py` - Engine orchestration
- `onnx_npu_engine.py` - Direct ONNX inference
- `model_converter.py` - Model conversion utils
- `main.py` - Ollama version
- `main_onnx.py` - ONNX version

Tapi **gunakan JavaScript version** (`main.js`) untuk hasil terbaik!

## 🎯 Perbandingan

### Python Version

```bash
# Complex setup
pip install -r requirements.txt
pip install optimum[onnxruntime]

# Manual model export
optimum-cli export onnx --model gpt2 ./models/gpt2_onnx

# Run with path
python src/main_onnx.py ./models/gpt2_onnx/model.onnx
```

### JavaScript Version

```bash
# Simple setup
npm install

# Auto model download
npm start
```

**JavaScript wins! 🏆**

## 🔄 Migration Steps (if you used Python before)

1. **Remove Python dependencies (optional):**

   ```bash
   # Clean Python venv if you want
   rm -rf .venv
   ```

2. **Install Node.js dependencies:**

   ```bash
   npm install
   ```

3. **Run JavaScript version:**

   ```bash
   npm start
   ```

4. **Enjoy!** Model downloads automatically, NPU works out of the box.

## 💡 Why We Switched

- Python ONNX export is complex
- NPU provider setup is manual
- JavaScript `@huggingface/transformers` handles everything
- Better developer experience
- Same or better performance

## 🎉 Result

Sekarang Anda bisa langsung pakai NPU dengan DeepSeek model tanpa ribet! 🚀

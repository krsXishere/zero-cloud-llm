# Quick Start Guide - JavaScript Version

## ⚡ Fastest Way to Get NPU Acceleration

### 1. Install Dependencies

```bash
npm install
```

### 2. Run the Application

**Interactive mode:**

```bash
npm start
```

**Demo mode:**

```bash
node src/main.js --demo
```

**Single prompt:**

```bash
node src/main.js --prompt "What is AI?"
```

### 3. First Run

The model will download automatically (~1GB):

```
🔄 Loading model...
⚠️  First run will download the model (~1GB). Please wait...
```

Wait for:

```
✅ Model loaded successfully!
🚀 NPU acceleration active (if available on your hardware)
```

### 4. Test It

```bash
npm test
```

This will:

- Load the DeepSeek model
- Run a math equation prompt
- Show streaming output
- Verify everything works

## 🎯 Expected Output

```
============================================================
ZERO CLOUD LLM - NPU Accelerated Inference
============================================================
Model: onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX
Quantization: q4f16
Max Tokens: 512
============================================================

🔄 Loading model...
✅ Model loaded successfully!
🚀 NPU acceleration active (if available on your hardware)

============================================================
INTERACTIVE MODE
============================================================
Commands:
  - Type your prompt and press Enter
  - Type 'quit' or 'exit' to exit
  - Type 'clear' to clear screen
============================================================

👤 You: _
```

## 🔍 Verify NPU Usage

1. **Open Task Manager** (Windows)
2. **Run the app** and send a prompt
3. **Watch CPU usage**:
   - With NPU: CPU stays low (<20%)
   - Without NPU: CPU spikes (50-100%)

## 🚀 Why JavaScript Version is Better

### Python Version

❌ Complex setup (ONNX export, providers, etc.)  
❌ Manual model conversion needed  
❌ NPU provider configuration  
❌ More code to write

### JavaScript Version

✅ One command: `npm install`  
✅ Auto-downloads ONNX models  
✅ Auto-detects NPU  
✅ Built-in streaming  
✅ Works out of the box

## 📝 Example Usage

```javascript
import { pipeline, TextStreamer } from "@huggingface/transformers";

// That's it! NPU is automatically used if available
const generator = await pipeline(
  "text-generation",
  "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX",
  { dtype: "q4f16" }
);

const messages = [{ role: "user", content: "Hello!" }];
const output = await generator(messages, { max_new_tokens: 100 });
```

## 🎨 Try Different Models

Edit `src/main.js`:

```javascript
// Smaller & faster
model: "onnx-community/Qwen2.5-0.5B-Instruct";

// Better quality
model: "onnx-community/Phi-3-mini-4k-instruct-onnx";

// Original (default)
model: "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX";
```

## 🐛 Common Issues

**"Cannot find module '@huggingface/transformers'"**

```bash
npm install
```

**Out of memory**

```javascript
// Use smaller model
model: "onnx-community/Qwen2.5-0.5B-Instruct";
```

**Slow on first run**

- Model is downloading (~1GB)
- Subsequent runs will be instant

## 🎉 That's It!

You now have a working NPU-accelerated LLM running locally!

No complex ONNX exports, no provider configuration, just works! 🚀

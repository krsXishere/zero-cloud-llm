# Troubleshooting Guide

## Error: "TypeError: Tensor.data must be a typed array (4) for float16 tensors"

### Penyebab

Error ini terjadi ketika hardware Anda tidak mendukung operasi float16 (16-bit floating point).

### Solusi ✅

Saya sudah memperbaiki `src/main.js` dengan **auto-fallback mechanism**:

1. **Pertama mencoba**: q4 (4-bit quantized)
2. **Fallback ke**: q8 (8-bit quantized)
3. **Fallback terakhir**: fp32 (32-bit float - paling kompatibel)

### Testing

Jalankan diagnostic script:

```bash
npm run diagnostic
```

Script ini akan:

- ✅ Check system info
- ✅ Test model loading dengan model kecil
- ✅ Memberikan rekomendasi

### Manual Fix

Jika masih error, edit `src/main.js`:

```javascript
// Ubah model ke yang lebih kompatibel
const CONFIG = {
  model: "Xenova/gpt2", // Paling kompatibel
  // atau
  model: "onnx-community/Qwen2.5-0.5B-Instruct", // Lebih kecil

  dtype: "fp32", // Paling stabil
  maxTokens: 256, // Kurangi jika out of memory
};
```

### Rekomendasi Model Berdasarkan Hardware

#### Low-end Hardware (4GB RAM)

```javascript
model: "Xenova/gpt2"; // ~500MB, sangat kompatibel
```

#### Mid-range Hardware (8GB RAM)

```javascript
model: "onnx-community/Qwen2.5-0.5B-Instruct"; // ~1GB, cepat
```

#### High-end Hardware (16GB+ RAM, NPU)

```javascript
model: "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX"; // ~1.5GB, kualitas bagus
```

## Error: Out of Memory

### Solusi

```javascript
const CONFIG = {
  model: "Xenova/gpt2", // Model lebih kecil
  maxTokens: 128, // Kurangi output tokens
};
```

## Error: Model Download Stuck/Slow

### Solusi 1: Set Cache Directory

```bash
# Mac/Linux
export HF_HOME="/path/to/large/drive/.cache/huggingface"

# Windows PowerShell
$env:HF_HOME="D:\.cache\huggingface"
```

### Solusi 2: Pre-download Model

```bash
# Install Hugging Face CLI
npm install -g @huggingface/hub

# Download model
huggingface-cli download onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX
```

## Error: Cannot find module '@huggingface/transformers'

### Solusi

```bash
rm -rf node_modules package-lock.json
npm install
```

## Verifikasi NPU Usage

### Windows Task Manager

1. Buka Task Manager (Ctrl+Shift+Esc)
2. Jalankan `npm start`
3. Kirim prompt
4. Amati:
   - **CPU**: Harus rendah (<20%) jika NPU aktif
   - **RAM**: Harus stabil, tidak naik terus
   - **NPU**: Harus ada aktivitas (jika ditampilkan)

### Mac Activity Monitor

1. Buka Activity Monitor
2. Jalankan `npm start`
3. Monitor "node" process
4. CPU usage rendah = NPU/GPU aktif

## Quick Diagnostic Commands

```bash
# Test basic functionality
npm run diagnostic

# Test with simple model
node src/test.js

# Check Node version (need 18+)
node --version

# Check installed packages
npm list @huggingface/transformers

# Clear cache and retry
rm -rf ~/.cache/huggingface
npm start
```

## Common Issues Summary

| Error             | Quick Fix                                |
| ----------------- | ---------------------------------------- |
| float16 error     | ✅ Already fixed - auto-fallback to fp32 |
| Out of memory     | Use smaller model: `Xenova/gpt2`         |
| Slow download     | Set `HF_HOME` to faster drive            |
| Can't find module | `npm install`                            |
| Stuck loading     | Check internet, clear cache              |

## Still Having Issues?

1. **Run diagnostic**: `npm run diagnostic`
2. **Try smallest model**: Change to `Xenova/gpt2` in `src/main.js`
3. **Check logs**: Look for specific error messages
4. **Update dependencies**: `npm update`
5. **Reinstall**: `rm -rf node_modules && npm install`

## Contact

Open an issue on GitHub with:

- Error message
- Output from `npm run diagnostic`
- Your OS and Node.js version
- Hardware specs (RAM, CPU, NPU)

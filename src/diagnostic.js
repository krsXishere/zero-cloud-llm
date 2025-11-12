/**
 * Diagnostic script for Zero Cloud LLM
 * Helps identify hardware capabilities and compatible models
 */

import { env } from "@huggingface/transformers";

console.log("🔍 Zero Cloud LLM - System Diagnostics\n");
console.log("=".repeat(70));

// Check environment
console.log("📋 Environment Information:");
console.log("=".repeat(70));
console.log(`Node.js version: ${process.version}`);
console.log(`Platform: ${process.platform}`);
console.log(`Architecture: ${process.arch}`);
console.log("");

// Check Transformers.js environment
console.log("=".repeat(70));
console.log("🤖 Transformers.js Configuration:");
console.log("=".repeat(70));

try {
    console.log(`Cache directory: ${env.cacheDir}`);
    console.log(`Allow remote models: ${env.allowRemoteModels}`);
    console.log(`Allow local models: ${env.allowLocalModels}`);
    console.log("");
} catch (error) {
    console.log("⚠️  Could not read environment settings");
    console.log("");
}

// Recommendations
console.log("=".repeat(70));
console.log("💡 Troubleshooting Tips:");
console.log("=".repeat(70));
console.log("1. If you see 'float16 tensor' errors:");
console.log("   → Your hardware may not support float16");
console.log("   → The updated main.js will auto-fallback to fp32");
console.log("");
console.log("2. If model download is slow:");
console.log("   → Set HF_HUB_CACHE environment variable");
console.log("   → Use a different mirror or VPN");
console.log("");
console.log("3. If out of memory:");
console.log("   → Try smaller model: Qwen2.5-0.5B-Instruct");
console.log("   → Reduce maxTokens in CONFIG");
console.log("");
console.log("4. Compatible models to try:");
console.log("   ✓ Xenova/gpt2 (most compatible, small)");
console.log("   ✓ onnx-community/Qwen2.5-0.5B-Instruct");
console.log("   ✓ onnx-community/Phi-3-mini-4k-instruct-onnx");
console.log("");

console.log("=".repeat(70));
console.log("🧪 Testing Model Load:");
console.log("=".repeat(70));

async function testModelLoad() {
    const { pipeline } = await import("@huggingface/transformers");

    // Test with most compatible model
    const testModel = "Xenova/gpt2";
    console.log(`\nTrying to load: ${testModel}`);
    console.log("This is a small, highly compatible model for testing...\n");

    try {
        console.log("Loading...");
        const generator = await pipeline(
            "text-generation",
            testModel,
            { dtype: "fp32" }
        );

        console.log("✅ SUCCESS! Your system can load ONNX models.");
        console.log("\nRunning test inference...");

        const output = await generator("Hello, ", {
            max_new_tokens: 10,
            do_sample: false,
        });

        console.log(`Output: "${output[0].generated_text}"`);
        console.log("\n✅ Inference works! You can now use main.js");

    } catch (error) {
        console.error("\n❌ Test failed:", error.message);
        console.error("\nPlease check:");
        console.error("1. Internet connection (for model download)");
        console.error("2. Disk space (~500MB needed)");
        console.error("3. Node.js version (need 18+)");
        console.error("\nFull error:");
        console.error(error);
    }
}

testModelLoad();

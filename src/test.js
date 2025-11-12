/**
 * Simple test script for Zero Cloud LLM
 */

import { pipeline, TextStreamer } from "@huggingface/transformers";

console.log("🧪 Testing ONNX DeepSeek Model...\n");

// Create a text generation pipeline
console.log("📦 Loading model...");
const generator = await pipeline(
    "text-generation",
    "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX",
    { dtype: "q4f16" }
);

console.log("✅ Model loaded!\n");

// Define the test message
const messages = [
    { role: "user", content: "Solve the equation: x^2 - 3x + 2 = 0" },
];

console.log("🤖 Generating response...\n");

// Create text streamer
const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
});

// Generate a response
const output = await generator(messages, {
    max_new_tokens: 512,
    do_sample: false,
    streamer,
});

console.log("\n\n✅ Test completed!");
console.log("📄 Full response:", output[0].generated_text.at(-1).content);

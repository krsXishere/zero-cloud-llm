/**
 * Zero Cloud LLM - Main Application
 * Uses ONNX DeepSeek model with NPU acceleration
 */

import { pipeline, TextStreamer } from "@huggingface/transformers";
import { createInterface } from "readline";

// Configuration
const CONFIG = {
    model: "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX",
    dtype: "q4f16", // Quantized for better performance
    maxTokens: 512,
    temperature: 0.7,
};

// ANSI color codes for better UI
const colors = {
    reset: "\x1b[0m",
    bright: "\x1b[1m",
    green: "\x1b[32m",
    blue: "\x1b[34m",
    yellow: "\x1b[33m",
    cyan: "\x1b[36m",
    red: "\x1b[31m",
};

/**
 * Display startup banner
 */
function displayBanner() {
    console.log("\n" + "=".repeat(70));
    console.log(
        colors.bright + colors.cyan + "ZERO CLOUD LLM - NPU Accelerated Inference" + colors.reset
    );
    console.log("=".repeat(70));
    console.log(`${colors.yellow}Model:${colors.reset} ${CONFIG.model}`);
    console.log(`${colors.yellow}Quantization:${colors.reset} ${CONFIG.dtype}`);
    console.log(`${colors.yellow}Max Tokens:${colors.reset} ${CONFIG.maxTokens}`);
    console.log("=".repeat(70) + "\n");
}

/**
 * Initialize the text generation pipeline
 */
async function initializePipeline() {
    console.log(colors.blue + "🔄 Loading model..." + colors.reset);
    console.log(
        colors.yellow +
        "⚠️  First run will download the model (~1GB). Please wait..." +
        colors.reset
    );
    console.log("");

    try {
        const generator = await pipeline(
            "text-generation",
            CONFIG.model,
            { dtype: CONFIG.dtype }
        );

        console.log(colors.green + "✅ Model loaded successfully!" + colors.reset);
        console.log(
            colors.cyan +
            "🚀 NPU acceleration active (if available on your hardware)" +
            colors.reset
        );
        console.log("");

        return generator;
    } catch (error) {
        console.error(colors.red + "❌ Error loading model:" + colors.reset, error.message);
        throw error;
    }
}

/**
 * Generate response for a single prompt
 */
async function generateResponse(generator, prompt, streaming = true) {
    const messages = [{ role: "user", content: prompt }];

    try {
        if (streaming) {
            // Create text streamer for real-time output
            const streamer = new TextStreamer(generator.tokenizer, {
                skip_prompt: true,
            });

            console.log(colors.green + "🤖 Assistant: " + colors.reset);

            const output = await generator(messages, {
                max_new_tokens: CONFIG.maxTokens,
                do_sample: false,
                streamer,
            });

            console.log("\n");
            return output[0].generated_text.at(-1).content;
        } else {
            // Non-streaming mode
            const output = await generator(messages, {
                max_new_tokens: CONFIG.maxTokens,
                do_sample: false,
            });

            const response = output[0].generated_text.at(-1).content;
            console.log(colors.green + "🤖 Assistant: " + colors.reset + response);
            console.log("");

            return response;
        }
    } catch (error) {
        console.error(colors.red + "❌ Error during generation:" + colors.reset, error.message);
        throw error;
    }
}

/**
 * Run demo mode with predefined prompts
 */
async function runDemoMode(generator) {
    const demos = [
        "Solve the equation: x^2 - 3x + 2 = 0",
        "What is the capital of Indonesia?",
        "Explain quantum computing in simple terms",
    ];

    console.log("\n" + "=".repeat(70));
    console.log(colors.bright + "DEMO MODE - Running Sample Prompts" + colors.reset);
    console.log("=".repeat(70) + "\n");

    for (let i = 0; i < demos.length; i++) {
        console.log(
            colors.cyan + `[Demo ${i + 1}/${demos.length}]` + colors.reset
        );
        console.log(colors.blue + "👤 User: " + colors.reset + demos[i]);
        console.log("");

        await generateResponse(generator, demos[i]);

        if (i < demos.length - 1) {
            console.log("-".repeat(70) + "\n");
        }
    }
}

/**
 * Run interactive mode
 */
async function runInteractiveMode(generator) {
    console.log("=".repeat(70));
    console.log(colors.bright + "INTERACTIVE MODE" + colors.reset);
    console.log("=".repeat(70));
    console.log("Commands:");
    console.log("  - Type your prompt and press Enter");
    console.log("  - Type 'quit' or 'exit' to exit");
    console.log("  - Type 'clear' to clear screen");
    console.log("=".repeat(70) + "\n");

    const rl = createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    const askQuestion = () => {
        rl.question(colors.blue + "👤 You: " + colors.reset, async (input) => {
            const userInput = input.trim();

            if (!userInput) {
                askQuestion();
                return;
            }

            if (["quit", "exit", "q"].includes(userInput.toLowerCase())) {
                console.log("\n" + colors.cyan + "👋 Goodbye!" + colors.reset + "\n");
                rl.close();
                return;
            }

            if (userInput.toLowerCase() === "clear") {
                console.clear();
                displayBanner();
                askQuestion();
                return;
            }

            console.log("");

            try {
                await generateResponse(generator, userInput);
            } catch (error) {
                console.error(colors.red + "Error:" + colors.reset, error.message);
            }

            askQuestion();
        });
    };

    askQuestion();
}

/**
 * Run single prompt mode
 */
async function runSinglePrompt(generator, prompt) {
    console.log(colors.blue + "👤 User: " + colors.reset + prompt);
    console.log("");
    await generateResponse(generator, prompt);
}

/**
 * Main application entry point
 */
async function main() {
    try {
        displayBanner();

        // Initialize pipeline
        const generator = await initializePipeline();

        // Parse command line arguments
        const args = process.argv.slice(2);

        if (args.length === 0) {
            // Default to interactive mode
            await runInteractiveMode(generator);
        } else if (args[0] === "--demo") {
            // Run demo mode
            await runDemoMode(generator);
        } else if (args[0] === "--prompt") {
            // Run single prompt mode
            if (args.length < 2) {
                console.error(colors.red + "Error: --prompt requires a prompt argument" + colors.reset);
                console.log("\nUsage:");
                console.log("  node src/main.js                    # Interactive mode");
                console.log("  node src/main.js --demo             # Demo mode");
                console.log('  node src/main.js --prompt "text"    # Single prompt');
                process.exit(1);
            }

            const prompt = args.slice(1).join(" ");
            await runSinglePrompt(generator, prompt);
        } else {
            console.error(colors.red + "Unknown argument: " + args[0] + colors.reset);
            console.log("\nUsage:");
            console.log("  node src/main.js                    # Interactive mode");
            console.log("  node src/main.js --demo             # Demo mode");
            console.log('  node src/main.js --prompt "text"    # Single prompt');
            process.exit(1);
        }
    } catch (error) {
        console.error(colors.red + "\n❌ Fatal error:" + colors.reset, error);
        process.exit(1);
    }
}

// Run the application
main();

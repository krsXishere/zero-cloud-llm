"""
Main Application Entry Point
Demonstrates usage of the NPU-accelerated LLM inference engine.
"""

import logging
import sys
from inference_engine import InferenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def interactive_mode(engine: InferenceEngine):
    """
    Run the engine in interactive mode.
    
    Args:
        engine: Initialized InferenceEngine instance
    """
    print("\n" + "="*60)
    print("Interactive Mode - NPU-Accelerated LLM")
    print("="*60)
    print("Commands:")
    print("  - Type your prompt and press Enter")
    print("  - Type 'quit' or 'exit' to exit")
    print("  - Type 'status' to see NPU status")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            if user_input.lower() == 'status':
                status = engine.get_npu_status()
                print("\n" + "="*60)
                print("NPU Status:")
                for key, value in status.items():
                    print(f"  {key}: {value}")
                print("="*60)
                continue
            
            # Generate response in streaming mode
            print("\n🤖 Assistant: ", end="", flush=True)
            
            for chunk in engine.generate_streaming(
                prompt=user_input,
                temperature=0.7,
                show_prompt=False
            ):
                print(chunk, end="", flush=True)
            
            print()  # New line after response
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break
        except Exception as e:
            logger.error(f"Error during interaction: {e}")
            print(f"\n❌ Error: {e}\n")


def demo_mode(engine: InferenceEngine):
    """
    Run a simple demo with predefined prompts.
    
    Args:
        engine: Initialized InferenceEngine instance
    """
    demo_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming."
    ]
    
    print("\n" + "="*60)
    print("Demo Mode - Running Sample Prompts")
    print("="*60)
    
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n[Demo {i}/{len(demo_prompts)}]")
        
        # Generate and print response
        for chunk in engine.generate_streaming(prompt=prompt):
            print(chunk, end="", flush=True)
        
        print("\n")  # Add spacing between demos


def main():
    """Main application entry point."""
    
    # Configuration
    MODEL_NAME = "deepseek-r1:1.5b"  # Change this to your preferred model
    OLLAMA_URL = "http://localhost:11434"
    
    # Create inference engine
    engine = InferenceEngine(
        model_name=MODEL_NAME,
        ollama_url=OLLAMA_URL
    )
    
    # Initialize engine
    if not engine.initialize():
        logger.error("Failed to initialize inference engine")
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            demo_mode(engine)
        elif sys.argv[1] == "--prompt":
            if len(sys.argv) < 3:
                print("Error: --prompt requires a prompt argument")
                sys.exit(1)
            
            prompt = " ".join(sys.argv[2:])
            print(f"\nPrompt: {prompt}\n")
            print("Response: ", end="", flush=True)
            
            for chunk in engine.generate_streaming(
                prompt=prompt,
                show_prompt=False
            ):
                print(chunk, end="", flush=True)
            print("\n")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("\nUsage:")
            print("  python main.py           # Interactive mode")
            print("  python main.py --demo    # Demo mode")
            print("  python main.py --prompt <your prompt>  # Single prompt")
            sys.exit(1)
    else:
        # Default to interactive mode
        interactive_mode(engine)


if __name__ == "__main__":
    main()

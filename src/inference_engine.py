"""
Inference Engine Module
Combines NPU acceleration with Ollama model inference.
"""

import logging
from typing import Optional, Generator
from .npu_detector import NPUDetector
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Main inference engine that combines NPU acceleration detection
    with Ollama model inference.
    """
    
    def __init__(
        self, 
        model_name: str = "deepseek-r1:1.5b",
        ollama_url: str = "http://localhost:11434"
    ):
        """
        Initialize inference engine.
        
        Args:
            model_name: Name of the Ollama model to use
            ollama_url: URL of the Ollama server
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        
        # Initialize components
        self.npu_detector = NPUDetector()
        self.ollama_client = OllamaClient(base_url=ollama_url, model=model_name)
        
        self.initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize the inference engine.
        Detects NPU availability and checks Ollama connection.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        logger.info("Initializing Inference Engine...")
        logger.info(f"Target Model: {self.model_name}")
        logger.info(f"Ollama URL: {self.ollama_url}")
        
        # Step 1: Detect NPU
        logger.info("\n[Step 1/2] Detecting NPU acceleration...")
        self.npu_detector.detect_npu()
        print(self.npu_detector.get_status_report())
        
        # Step 2: Check Ollama connection
        logger.info("\n[Step 2/2] Connecting to Ollama server...")
        if not self.ollama_client.check_connection():
            logger.error("Failed to connect to Ollama server")
            return False
        
        # List available models
        models = self.ollama_client.list_models()
        if models:
            model_names = [m.get("name", "unknown") for m in models]
            logger.info(f"Available models: {', '.join(model_names)}")
            
            # Check if target model is available
            if self.model_name not in model_names:
                logger.warning(f"Model '{self.model_name}' not found in available models")
                logger.warning(f"You may need to pull it first: ollama pull {self.model_name}")
        
        self.initialized = True
        logger.info("\n✓ Inference Engine initialized successfully\n")
        return True
    
    def generate_streaming(
        self, 
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        show_prompt: bool = True
    ) -> Generator[str, None, None]:
        """
        Generate text from model in streaming mode.
        
        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            show_prompt: Whether to print the prompt before generation
            
        Yields:
            str: Generated text chunks
        """
        if not self.initialized:
            logger.error("Inference engine not initialized. Call initialize() first.")
            yield "Error: Engine not initialized\n"
            return
        
        if show_prompt:
            print(f"\n{'='*60}")
            print(f"PROMPT:")
            print(f"{'='*60}")
            print(prompt)
            print(f"{'='*60}")
            print(f"RESPONSE (using {self.npu_detector.selected_provider}):")
            print(f"{'='*60}")
        
        # NOTE: Currently Ollama handles the inference internally.
        # NPU acceleration would be used if:
        # 1. The model is exported to ONNX/QNN format
        # 2. We load it with onnxruntime using the detected NPU provider
        # 
        # For now, we're using Ollama's API which may or may not use NPU
        # depending on Ollama's internal implementation.
        # The infrastructure is ready for when we convert models to ONNX.
        
        for chunk in self.ollama_client.generate_streaming(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        ):
            yield chunk
    
    def generate(
        self, 
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from model (non-streaming).
        
        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Generated text
        """
        result = ""
        for chunk in self.generate_streaming(
            prompt=prompt, 
            temperature=temperature,
            max_tokens=max_tokens,
            show_prompt=False
        ):
            result += chunk
        return result
    
    def get_npu_status(self) -> dict:
        """
        Get current NPU acceleration status.
        
        Returns:
            dict: Status information
        """
        return {
            "npu_available": self.npu_detector.npu_available,
            "selected_provider": self.npu_detector.selected_provider,
            "available_providers": self.npu_detector.available_providers,
            "onnxruntime_available": self.npu_detector.onnxruntime_available
        }

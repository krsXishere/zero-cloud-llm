"""
Ollama Client Module
Handles communication with local Ollama API for model inference.
"""

import json
import logging
import requests
from typing import Generator, Dict, Any, Optional

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with local Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-r1:1.5b"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Base URL for Ollama API
            model: Model name to use for inference
        """
        self.base_url = base_url
        self.model = model
        self.generate_endpoint = f"{base_url}/api/generate"
        self.tags_endpoint = f"{base_url}/api/tags"
        
    def check_connection(self) -> bool:
        """
        Check if Ollama server is reachable.
        
        Returns:
            bool: True if server is reachable, False otherwise
        """
        try:
            response = requests.get(self.tags_endpoint, timeout=5)
            if response.status_code == 200:
                logger.info("✓ Successfully connected to Ollama server")
                return True
            else:
                logger.error(f"✗ Ollama server returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Failed to connect to Ollama server: {e}")
            return False
    
    def list_models(self) -> Optional[list]:
        """
        List available models in Ollama.
        
        Returns:
            Optional[list]: List of model information, or None if request fails
        """
        try:
            response = requests.get(self.tags_endpoint, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                logger.info(f"Found {len(models)} available model(s)")
                return models
            return None
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return None
    
    def generate_streaming(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        Generate text from model in streaming mode.
        
        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Yields:
            str: Generated text chunks
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            logger.info(f"Sending prompt to model '{self.model}'...")
            
            with requests.post(
                self.generate_endpoint, 
                json=payload, 
                stream=True,
                timeout=60
            ) as response:
                
                if response.status_code != 200:
                    error_msg = f"API returned status code {response.status_code}"
                    logger.error(error_msg)
                    yield f"\nError: {error_msg}\n"
                    return
                
                # Process streaming JSON responses
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            
                            # Extract the generated text
                            if "response" in chunk:
                                yield chunk["response"]
                            
                            # Check if generation is complete
                            if chunk.get("done", False):
                                logger.info("Generation complete")
                                break
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to decode JSON chunk: {e}")
                            continue
                            
        except requests.exceptions.Timeout:
            error_msg = "Request timed out"
            logger.error(error_msg)
            yield f"\nError: {error_msg}\n"
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {e}"
            logger.error(error_msg)
            yield f"\nError: {error_msg}\n"
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(error_msg)
            yield f"\nError: {error_msg}\n"
    
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
        for chunk in self.generate_streaming(prompt, temperature, max_tokens):
            result += chunk
        return result

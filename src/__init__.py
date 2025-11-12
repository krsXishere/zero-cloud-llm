"""
Zero Cloud LLM - Local LLM with NPU Acceleration
"""

from .npu_detector import NPUDetector
from .ollama_client import OllamaClient
from .inference_engine import InferenceEngine

__all__ = [
    "NPUDetector",
    "OllamaClient", 
    "InferenceEngine"
]

__version__ = "0.1.0"

"""
NPU Detection Module
Detects and initializes Qualcomm NPU acceleration for inference.
"""

import logging
from typing import Optional, List
import sys

logger = logging.getLogger(__name__)


class NPUDetector:
    """Detects and manages NPU acceleration availability."""
    
    def __init__(self):
        self.available_providers: List[str] = []
        self.npu_available: bool = False
        self.selected_provider: str = "CPUExecutionProvider"
        self.onnxruntime_available: bool = False
        
    def detect_npu(self) -> bool:
        """
        Detects if Qualcomm NPU is available for acceleration.
        
        Returns:
            bool: True if NPU is available, False otherwise
        """
        try:
            import onnxruntime as ort
            self.onnxruntime_available = True
            
            self.available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX Runtime providers: {self.available_providers}")
            
            # Check for Qualcomm NPU providers
            # Common NPU provider names:
            # - QNNExecutionProvider (Qualcomm Neural Network SDK)
            # - SNPEExecutionProvider (Snapdragon Neural Processing Engine)
            # - DMLExecutionProvider (DirectML - may use NPU on Windows on ARM)
            npu_providers = [
                "QNNExecutionProvider",
                "SNPEExecutionProvider", 
                "DMLExecutionProvider"
            ]
            
            for provider in npu_providers:
                if provider in self.available_providers:
                    self.npu_available = True
                    self.selected_provider = provider
                    logger.info(f"✓ NPU acceleration available via {provider}")
                    return True
            
            logger.warning("✗ No NPU provider found. Falling back to CPU.")
            self.selected_provider = "CPUExecutionProvider"
            return False
            
        except ImportError:
            logger.warning("onnxruntime not installed. NPU detection skipped.")
            self.onnxruntime_available = False
            return False
        except Exception as e:
            logger.error(f"Error during NPU detection: {e}")
            return False
    
    def get_execution_providers(self) -> List[str]:
        """
        Returns the list of execution providers to use.
        NPU provider first if available, then CPU as fallback.
        
        Returns:
            List[str]: Ordered list of execution providers
        """
        if self.npu_available:
            # Return NPU provider with CPU as fallback
            return [self.selected_provider, "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    
    def get_status_report(self) -> str:
        """
        Returns a human-readable status report.
        
        Returns:
            str: Status report string
        """
        status = []
        status.append("=" * 60)
        status.append("NPU Acceleration Status")
        status.append("=" * 60)
        status.append(f"ONNX Runtime Available: {self.onnxruntime_available}")
        status.append(f"NPU Available: {self.npu_available}")
        status.append(f"Selected Provider: {self.selected_provider}")
        status.append(f"All Available Providers: {', '.join(self.available_providers) if self.available_providers else 'None'}")
        status.append("=" * 60)
        return "\n".join(status)

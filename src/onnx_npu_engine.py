"""
ONNX NPU Inference Engine
Direct model inference using ONNX Runtime with NPU acceleration.
"""

import logging
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ONNXNPUEngine:
    """
    Engine for running ONNX models directly on NPU.
    This bypasses Ollama and uses onnxruntime directly.
    """
    
    def __init__(self, model_path: str, npu_provider: str = "QNNExecutionProvider"):
        """
        Initialize ONNX NPU Engine.
        
        Args:
            model_path: Path to ONNX model file
            npu_provider: NPU execution provider name
        """
        self.model_path = Path(model_path)
        self.npu_provider = npu_provider
        self.session = None
        self.input_names = []
        self.output_names = []
        self.is_npu = False
        
    def load_model(self) -> bool:
        """
        Load ONNX model with NPU acceleration.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            import onnxruntime as ort
            
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            logger.info(f"Loading ONNX model: {self.model_path}")
            
            # Get available providers
            available_providers = ort.get_available_providers()
            logger.info(f"Available providers: {available_providers}")
            
            # Setup execution providers
            providers = []
            provider_options = []
            
            # Try to use NPU provider
            if self.npu_provider in available_providers:
                logger.info(f"✓ Using NPU provider: {self.npu_provider}")
                
                # Configure provider options based on type
                if self.npu_provider == "QNNExecutionProvider":
                    qnn_options = {
                        'backend_path': 'QnnHtp.dll',  # For Hexagon DSP/NPU
                        'qnn_context_priority': 'high',
                        'htp_performance_mode': 'burst',  # Maximum performance
                    }
                    providers.append(self.npu_provider)
                    provider_options.append(qnn_options)
                    
                elif self.npu_provider == "DMLExecutionProvider":
                    dml_options = {
                        'device_id': 0,
                    }
                    providers.append(self.npu_provider)
                    provider_options.append(dml_options)
                    
                else:
                    providers.append(self.npu_provider)
                    provider_options.append({})
                
                self.is_npu = True
                
            else:
                logger.warning(f"NPU provider '{self.npu_provider}' not available")
                logger.warning("Falling back to CPU")
            
            # Always add CPU as fallback
            providers.append("CPUExecutionProvider")
            provider_options.append({})
            
            # Create session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create inference session
            logger.info(f"Creating session with providers: {providers}")
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options
            )
            
            # Get actual provider used
            actual_provider = self.session.get_providers()[0]
            logger.info(f"✓ Session created with provider: {actual_provider}")
            
            if actual_provider != "CPUExecutionProvider":
                logger.info("✓✓✓ NPU ACCELERATION ACTIVE ✓✓✓")
                self.is_npu = True
            else:
                logger.warning("Using CPU - NPU not active")
                self.is_npu = False
            
            # Get input/output names
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            logger.info(f"Model inputs: {self.input_names}")
            logger.info(f"Model outputs: {self.output_names}")
            
            return True
            
        except ImportError:
            logger.error("onnxruntime not installed. Please install: pip install onnxruntime")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference on NPU/CPU.
        
        Args:
            inputs: Dictionary of input name -> numpy array
            
        Returns:
            Dictionary of output name -> numpy array
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Run inference
            outputs = self.session.run(self.output_names, inputs)
            
            # Convert to dictionary
            return {name: output for name, output in zip(self.output_names, outputs)}
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current engine status.
        
        Returns:
            Dictionary with status information
        """
        if self.session:
            actual_provider = self.session.get_providers()[0]
        else:
            actual_provider = "Not loaded"
        
        return {
            "model_loaded": self.session is not None,
            "model_path": str(self.model_path),
            "npu_active": self.is_npu,
            "provider": actual_provider,
            "input_names": self.input_names,
            "output_names": self.output_names
        }

"""
Model Converter Utility
Converts models from various formats to ONNX for NPU acceleration.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ModelConverter:
    """
    Utility for converting models to ONNX format.
    """
    
    @staticmethod
    def check_ollama_model(model_name: str) -> bool:
        """
        Check if an Ollama model exists locally.
        
        Args:
            model_name: Name of the Ollama model
            
        Returns:
            bool: True if model exists, False otherwise
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return model_name in result.stdout
            return False
            
        except Exception as e:
            logger.error(f"Failed to check Ollama model: {e}")
            return False
    
    @staticmethod
    def get_ollama_model_path(model_name: str) -> Optional[str]:
        """
        Get the file path of an Ollama model.
        
        Args:
            model_name: Name of the Ollama model
            
        Returns:
            Optional[str]: Path to model file, or None if not found
        """
        # Ollama models are typically stored in:
        # Windows: %USERPROFILE%\.ollama\models
        # Linux/Mac: ~/.ollama/models
        
        import os
        import platform
        
        if platform.system() == "Windows":
            base_path = Path(os.environ.get("USERPROFILE", "")) / ".ollama" / "models"
        else:
            base_path = Path.home() / ".ollama" / "models"
        
        logger.info(f"Ollama models base path: {base_path}")
        
        if not base_path.exists():
            logger.warning(f"Ollama models directory not found: {base_path}")
            return None
        
        # Ollama stores models in subdirectories
        # The structure is complex, so we'll need to explore
        manifests_path = base_path / "manifests"
        blobs_path = base_path / "blobs"
        
        logger.info(f"Manifests path: {manifests_path}")
        logger.info(f"Blobs path: {blobs_path}")
        
        # This is a placeholder - actual implementation would need to:
        # 1. Parse manifest files
        # 2. Find the model weights blob
        # 3. Return the path to the weights
        
        return str(base_path)
    
    @staticmethod
    def convert_gguf_to_onnx(
        gguf_path: str,
        output_path: str,
        quantization: str = "int4"
    ) -> bool:
        """
        Convert GGUF model to ONNX format.
        
        This is a placeholder for actual conversion logic.
        You would typically use tools like:
        - optimum-cli (Hugging Face)
        - llama.cpp export
        - Custom conversion scripts
        
        Args:
            gguf_path: Path to GGUF model file
            output_path: Path for output ONNX file
            quantization: Quantization type (int4, int8, fp16)
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        logger.warning("GGUF to ONNX conversion not implemented yet")
        logger.info("To convert your model to ONNX:")
        logger.info("1. For Hugging Face models:")
        logger.info("   pip install optimum[onnxruntime]")
        logger.info("   optimum-cli export onnx --model <model_name> <output_dir>")
        logger.info("")
        logger.info("2. For GGUF models:")
        logger.info("   Use llama.cpp export functionality or convert via PyTorch")
        logger.info("")
        logger.info("3. For Ollama models:")
        logger.info("   Extract the model weights and convert using appropriate tools")
        
        return False
    
    @staticmethod
    def print_conversion_guide():
        """
        Print a guide for converting models to ONNX.
        """
        guide = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    MODEL CONVERSION GUIDE FOR NPU                          ║
╚════════════════════════════════════════════════════════════════════════════╝

To use NPU acceleration, you need to convert your model to ONNX format.

📦 OPTION 1: Convert Hugging Face Model to ONNX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Install Optimum:
   pip install optimum[onnxruntime]

2. Export model:
   optimum-cli export onnx --model deepseek-ai/deepseek-llm-7b-base ./onnx_model

3. Quantize for NPU (optional):
   python -m onnxruntime.quantization.quantize_dynamic \\
       --model_input model.onnx \\
       --model_output model_int8.onnx \\
       --per_channel \\
       --reduce_range

📦 OPTION 2: Use Pre-exported ONNX Models
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Look for models already in ONNX format:
- Hugging Face Hub: Search for "onnx" models
- Microsoft's optimized models: microsoft/onnx-models

📦 OPTION 3: Convert GGUF Models (Advanced)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This requires custom conversion scripts as GGUF → ONNX
is not straightforward for LLMs.

⚡ OPTION 4: Use Smaller ONNX-Compatible Models
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Start with smaller models that are easier to convert:
- BERT, DistilBERT (for embeddings/classification)
- Phi-2, Phi-3 (Microsoft's small LLMs)
- GPT-2 (classic small model)

Example with GPT-2:
   optimum-cli export onnx --model gpt2 ./gpt2_onnx

🔧 QUALCOMM-SPECIFIC: QNN Conversion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For best NPU performance on Qualcomm:
1. Convert to ONNX first (steps above)
2. Use Qualcomm AI Engine Direct SDK:
   - Install QNN SDK
   - Use qnn-onnx-converter tool
   - Optimize for Hexagon DSP/NPU

   qnn-onnx-converter --input_network model.onnx \\
                      --output_path model_qnn.cpp

For more info: https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk

═══════════════════════════════════════════════════════════════════════════════

💡 RECOMMENDED APPROACH FOR TESTING:
1. Start with a small ONNX model (GPT-2, DistilBERT)
2. Test NPU acceleration with our code
3. Once working, convert larger models

"""
        print(guide)

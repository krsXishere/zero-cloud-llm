"""
Main Application Entry Point - ONNX NPU Version
Demonstrates DIRECT NPU usage with ONNX models.
"""

import logging
import sys
import numpy as np
from pathlib import Path
from npu_detector import NPUDetector
from onnx_npu_engine import ONNXNPUEngine
from model_converter import ModelConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def test_npu_with_onnx(model_path: str):
    """
    Test NPU acceleration with an ONNX model.
    
    Args:
        model_path: Path to ONNX model file
    """
    print("\n" + "="*70)
    print("NPU ACCELERATION TEST - DIRECT ONNX INFERENCE")
    print("="*70)
    
    # Step 1: Detect NPU
    detector = NPUDetector()
    detector.detect_npu()
    print(detector.get_status_report())
    
    if not detector.npu_available:
        print("\n⚠️  WARNING: No NPU detected!")
        print("The model will run on CPU instead.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Step 2: Load model with NPU
    print(f"\n{'='*70}")
    print("LOADING MODEL WITH NPU ACCELERATION")
    print("="*70)
    
    engine = ONNXNPUEngine(
        model_path=model_path,
        npu_provider=detector.selected_provider
    )
    
    if not engine.load_model():
        logger.error("Failed to load model")
        return
    
    # Show status
    status = engine.get_status()
    print("\n" + "="*70)
    print("ENGINE STATUS")
    print("="*70)
    for key, value in status.items():
        print(f"  {key}: {value}")
    print("="*70)
    
    if status['npu_active']:
        print("\n✅✅✅ NPU IS ACTIVE - WILL NOT USE RAM/CPU/GPU HEAVILY ✅✅✅\n")
    else:
        print("\n⚠️  WARNING: NPU NOT ACTIVE - Will use CPU\n")
    
    # Step 3: Run test inference
    print("\n" + "="*70)
    print("RUNNING TEST INFERENCE")
    print("="*70)
    
    try:
        # Create dummy input (you'll need to adjust based on your model)
        # This is just an example - real LLM input would be different
        print("\nNote: Inference will depend on your model's input/output format")
        print(f"Expected inputs: {status['input_names']}")
        print(f"Expected outputs: {status['output_names']}")
        
        # Example: If your model expects input_ids
        if 'input_ids' in status['input_names']:
            # Create dummy token IDs (batch_size=1, seq_len=10)
            dummy_input = {
                'input_ids': np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int64)
            }
            
            print("\nRunning inference with dummy input...")
            outputs = engine.infer(dummy_input)
            
            print("\n✓ Inference completed successfully!")
            print(f"Output keys: {list(outputs.keys())}")
            for key, value in outputs.items():
                print(f"  {key} shape: {value.shape}")
        else:
            print("\n⚠️  Cannot run test inference - unknown input format")
            print("You'll need to create appropriate inputs for your model")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main application entry point."""
    
    print("\n" + "="*70)
    print("ZERO CLOUD LLM - DIRECT NPU ACCELERATION")
    print("="*70)
    
    if len(sys.argv) < 2:
        print("\n❌ Error: No model path provided\n")
        print("Usage:")
        print("  python main_onnx.py <path_to_onnx_model>")
        print("\nExample:")
        print("  python main_onnx.py models/model.onnx")
        print("\n" + "-"*70)
        print("Don't have an ONNX model yet? Here's how to get one:")
        print("-"*70)
        ModelConverter.print_conversion_guide()
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"\n❌ Error: Model file not found: {model_path}\n")
        sys.exit(1)
    
    # Run NPU test
    test_npu_with_onnx(model_path)


if __name__ == "__main__":
    main()

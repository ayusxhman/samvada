import sys
import os
from pathlib import Path
import torch

# Add root project directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.models.fingerspell_model import FingerspellCNN
from backend.app.config import FINGERSPELLING_MODEL_PATH, TRAINED_MODELS_DIR

def export_fingerspell_to_onnx():
    print("Starting ONNX export for Fingerspelling CNN model...")
    
    if not FINGERSPELLING_MODEL_PATH.exists():
        print(f"[FAIL] Checkpoint not found at {FINGERSPELLING_MODEL_PATH}")
        return False
        
    onnx_output_path = TRAINED_MODELS_DIR / "fingerspell_model.onnx"
    
    try:
        # 1. Load PyTorch model
        model = FingerspellCNN(num_classes=29)
        model.load_state_dict(torch.load(FINGERSPELLING_MODEL_PATH, map_location="cpu"))
        model.eval()
        print("[OK] PyTorch model weights loaded successfully.")

        # 2. Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # 3. Export to ONNX format using classic TorchScript exporter
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_output_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                },
                dynamo=False
            )
        except TypeError:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_output_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                }
            )
            
        print(f"[OK] Successfully exported ONNX model to {onnx_output_path}")
        print(f"[OK] ONNX File Size: {onnx_output_path.stat().st_size / (1024*1024):.2f} MB")
        return True
    except Exception as e:
        print(f"[FAIL] ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    export_fingerspell_to_onnx()

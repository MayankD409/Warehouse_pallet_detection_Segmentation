#!/usr/bin/env python3
from ultralytics import YOLO
import os
import onnx
import onnxsim
import argparse
import numpy as np
from pathlib import Path

def optimize_onnx_model(onnx_model_path, output_dir, quantize=True):
    """
    Optimize an ONNX model with simplification and optional quantization.
    
    Args:
        onnx_model_path: Path to the ONNX model
        output_dir: Directory to save the optimized model
        quantize: Whether to apply quantization
    
    Returns:
        Path to the optimized model
    """
    print(f"Optimizing ONNX model: {onnx_model_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    
    # Simplify the model
    model_simp, check = onnxsim.simplify(model)
    if not check:
        print("Simplified ONNX model could not be validated")
    else:
        print("Simplified ONNX model validated")
        model = model_simp
    
    # Get the filename without extension
    base_name = Path(onnx_model_path).stem
    
    # Save the simplified model
    optimized_path = os.path.join(output_dir, f"{base_name}_simplified.onnx")
    onnx.save(model, optimized_path)
    print(f"Saved simplified model to {optimized_path}")
    
    # Apply quantization if requested
    if quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantized_path = os.path.join(output_dir, f"{base_name}_quantized.onnx")
        quantize_dynamic(optimized_path, quantized_path, weight_type=QuantType.QUInt8)
        print(f"Saved quantized model to {quantized_path}")
        return quantized_path
    
    return optimized_path

def main():
    parser = argparse.ArgumentParser(description='ONNX Model Optimization')
    parser.add_argument('--detection_model', type=str, 
                       default=os.path.join("trained_models", "unoptimized", "detect", "best.pt"),
                       help='Path to the detection model')
    parser.add_argument('--segmentation_model', type=str, 
                       default=os.path.join("trained_models", "unoptimized", "segment", "best.pt"),
                       help='Path to the segmentation model')
    parser.add_argument('--output_dir', type=str, 
                       default="trained_models/optimized",
                       help='Directory to save optimized models')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply quantization to the model')
    parser.add_argument('--skip_export', action='store_true',
                       help='Skip PyTorch to ONNX export if ONNX already exists')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Detection model
    if os.path.exists(args.detection_model):
        print(f"Loading detection model from: {args.detection_model}")
        model_detect = YOLO(args.detection_model)
        
        # Export to ONNX if needed
        detect_onnx_path = os.path.join(args.output_dir, "detection.onnx")
        if args.skip_export and os.path.exists(detect_onnx_path):
            print(f"Using existing ONNX detection model: {detect_onnx_path}")
        else:
            print("Exporting detection model to ONNX...")
            detect_onnx_path = model_detect.export(format="onnx", imgsz=640, 
                                                 opset=12, simplify=True, 
                                                 output=detect_onnx_path)
        
        # Optimize the exported ONNX model
        optimize_onnx_model(detect_onnx_path, args.output_dir, args.quantize)
    else:
        print(f"Warning: Detection model not found at {args.detection_model}")
    
    # Segmentation model
    if os.path.exists(args.segmentation_model):
        print(f"Loading segmentation model from: {args.segmentation_model}")
        model_seg = YOLO(args.segmentation_model)
        
        # Export to ONNX if needed
        seg_onnx_path = os.path.join(args.output_dir, "segmentation.onnx")
        if args.skip_export and os.path.exists(seg_onnx_path):
            print(f"Using existing ONNX segmentation model: {seg_onnx_path}")
        else:
            print("Exporting segmentation model to ONNX...")
            seg_onnx_path = model_seg.export(format="onnx", imgsz=640, 
                                           opset=12, simplify=True, 
                                           output=seg_onnx_path)
        
        # Optimize the exported ONNX model
        optimize_onnx_model(seg_onnx_path, args.output_dir, args.quantize)
    else:
        print(f"Warning: Segmentation model not found at {args.segmentation_model}")

if __name__ == "__main__":
    main()
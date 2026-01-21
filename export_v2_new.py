import argparse
import cv2
import glob
import numpy as np
import os
import torch
import torch.onnx

from depth_anything_v2.dpt import DepthAnythingV2

def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--suffix', type=str, default='')

    args = parser.parse_args()
    
    # Model configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Initialize the model
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    from safetensors.torch import load_file
    # depth_anything.load_state_dict(load_file(r'C:\owl\3DEngine\models\model19.safetensors', device='cpu'))
    # depth_anything.load_state_dict(load_file(f'/home/hoiliu/julian/work/Depth-Anything-V2/checkpoints/Distill-Any-Depth-Multi-Teacher-Small.safetensors', device='cpu'))
    depth_anything.load_state_dict(load_file(f'/mnt/model-weights/depth-model/distill-any-depth-multi-teacher-small.safetensors', device='cpu'))
    
    # Convert model to specified precision
    device = 'cpu'
    dtype = torch.float32
    depth_anything = depth_anything.to(device).eval()

    # Define dummy input data in specified precision
    dummy_input = torch.ones((1, 3, args.input_size, args.input_size), dtype=dtype).to(device)

    # Provide an example input to the model
    example_output = depth_anything.forward(dummy_input)

    # Define ONNX export path
    # onnx_path = f'depth_anything_v2_{args.encoder}_{args.input_size}.onnx'
    onnx_path = f'distill_any_depth_multi_teacher_small_{args.encoder}_{args.input_size}_{args.suffix}.onnx'

    # Export the PyTorch model to ONNX format
    torch.onnx.export(
        depth_anything,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        verbose=True,
        export_params=True,
        do_constant_folding=True
    )

    print(f"model exported to {onnx_path}")

if __name__ == "__main__":
    main()
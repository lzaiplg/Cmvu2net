import os
import sys
import argparse
import yaml
import time
import torch
import torch.nn as nn
import numpy as np
import psutil  

current_dir = os.path.dirname(os.path.abspath(__file__))
pytorch_dir = os.path.join(current_dir, '../pytorch')
if pytorch_dir not in sys.path:
    sys.path.append(pytorch_dir)

try:
    from thop import profile
    from thop import clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

from models import get_model

def parse_args():
    parser = argparse.ArgumentParser(description='Model Evaluation on CPU (Params, FLOPs, FPS, Latency, Memory)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model weights')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for FPS/Latency test')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers (ignored for random input)')
    parser.add_argument('--iters', type=int, default=100, help='Number of iterations for timing')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--input_size', type=str, default='256,256', help='Input size H,W')
    parser.add_argument('--opts', nargs='+', help='Config overrides')
    return parser.parse_args()

def load_config(config_path, args):
    """Load and merge config from YAML and command line."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.opts:
        for i in range(0, len(args.opts), 2):
            key, value = args.opts[i], args.opts[i+1]
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            try:
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
            except AttributeError:
                pass
            d[keys[-1]] = value
    
    return config

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024

def main():
    args = parse_args()
    
    # Force using CPU
    device = torch.device('cpu')
    print("Using CPU for evaluation")

    # Load config
    config = load_config(args.config, args)
    
    # Input size
    try:
        h, w = map(int, args.input_size.split(','))
    except:
        h, w = 448, 448
        print(f"Invalid input size format, using default {h}x{w}")
    
    initial_memory = get_memory_usage()
    print(f"Initial Memory Usage: {initial_memory:.2f} MB")
    
    # Create model
    model_cfg = config.get('model', {}).copy()
    model_type = model_cfg.pop('type', 'Unknown')
    print(f"Creating model: {model_type}")
    model = get_model(model_type, **model_cfg)
    
    model_memory = get_memory_usage()
    print(f"Model Creation Memory Usage: {model_memory:.2f} MB")
    
    # Load weights if provided
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}")
        try:
            checkpoint = torch.load(args.model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict)
            print("Weights loaded successfully")
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print("No weights loaded or file not found (testing with random init)")

    model.to(device)
    model.eval()

    weight_memory = get_memory_usage()
    print(f"After Loading Weights Memory Usage: {weight_memory:.2f} MB")

    # Create dummy input
    input_tensor = torch.randn(args.batch_size, 3, h, w).to(device)
    
    print("-" * 50)
    print(f"Model: {model_type}")
    print(f"Input Size: {h}x{w}")
    print(f"Batch Size: {args.batch_size}")
    print("-" * 50)

    # 1. Measure Params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.3f} M")

    # 2. Measure FLOPs
    if HAS_THOP:
        try:
            # thop requires input to be on the same device as model
            # and sometimes expects specific input format
            flops_input = torch.randn(1, 3, h, w).to(device)
            flops, params = profile(model, inputs=(flops_input,), verbose=False)
            # flops, params = clever_format([flops, params], "%.3f")
            flops_g = flops / 1e9
            print(f"FLOPs: {flops_g:.3f} G")
            # print(f"Params (thop): {params}") # using model.parameters() instead
        except Exception as e:
            print(f"Error calculating FLOPs with thop: {e}")
    else:
        print("thop not installed, skipping FLOPs calculation (pip install thop)")

    # 3. Measure RSS Memory during inference
    print(f"Warming up ({args.warmup} iters)...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(input_tensor)
    
    warmup_memory = get_memory_usage()
    
    print(f"Running speed test ({args.iters} iters)...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(args.iters):
            _ = model(input_tensor)
    end_time = time.time()
    
    final_memory = get_memory_usage()
    
    total_time = end_time - start_time
    avg_time = total_time / args.iters
    fps = (args.iters * args.batch_size) / total_time
    latency_ms = (avg_time * 1000) / args.batch_size
    
    print("-" * 50)
    print(f"Total Time: {total_time:.4f} s")
    print(f"Average Time per Batch: {avg_time:.4f} s")
    print(f"FPS: {fps:.2f}")
    print(f"Latency: {latency_ms:.4f} ms/image")
    print(f"Initial Memory: {initial_memory:.2f} MB")
    print(f"Model Memory: {model_memory:.2f} MB")
    print(f"With Weights Memory: {weight_memory:.2f} MB")
    print(f"After Warmup Memory: {warmup_memory:.2f} MB")
    print(f"Final Memory: {final_memory:.2f} MB")
    print(f"Peak Memory Increase: {max(model_memory, weight_memory, warmup_memory, final_memory) - initial_memory:.2f} MB")
    print("-" * 50)

if __name__ == '__main__':
    main()
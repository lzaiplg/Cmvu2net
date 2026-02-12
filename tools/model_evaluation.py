import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np

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
    parser = argparse.ArgumentParser(description='Model Evaluation (Params, FLOPs, FPS, Latency)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model weights')
    parser.add_argument('--device', type=str, default='gpu', help='Device: gpu or cpu')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for FPS/Latency test')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers (ignored for random input)')
    parser.add_argument('--iters', type=int, default=100, help='Number of iterations for timing')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--input_size', type=str, default='256,256', help='Input size H,W')
    parser.add_argument('--opts', nargs='+', help='Config overrides')
    return parser.parse_args()

def load_config(config_path, args):
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

def main():
    args = parse_args()
    
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    config = load_config(args.config, args)
    
    try:
        h, w = map(int, args.input_size.split(','))
    except:
        h, w = 448, 448
        print(f"Invalid input size format, using default {h}x{w}")
    
    model_cfg = config.get('model', {}).copy()
    model_type = model_cfg.pop('type', 'Unknown')
    print(f"Creating model: {model_type}")
    model = get_model(model_type, **model_cfg)
    
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}")
        try:
            checkpoint = torch.load(args.model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
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
    
    print("-" * 50)
    print(f"Model: {model_type}")
    print(f"Input Size: {h}x{w}")
    print(f"Batch Size: {args.batch_size}")
    print("-" * 50)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.3f} M")

    if HAS_THOP:
        try:
            flops_input = torch.randn(1, 3, h, w).to(device)
            flops, params = profile(model, inputs=(flops_input,), verbose=False)
            flops_g = flops / 1e9
            print(f"FLOPs: {flops_g:.3f} G")
        except Exception as e:
            print(f"Error calculating FLOPs with thop: {e}")
    else:
        print("thop not installed, skipping FLOPs calculation (pip install thop)")

if __name__ == '__main__':
    main()

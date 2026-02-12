import os
import sys
import argparse
import yaml
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import subprocess
import glob
import shutil
import re
import random
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from models import get_model
from datasets import CrackDataset, Compose, Resize, RandomHorizontalFlip, RandomCrop, Normalize, BinaryLabelConvert, get_collate_fn, FourValueLabelConvert, ResizeStepScaling, RandomPaddingCrop, RandomDistort
from losses import WeightedCrossEntropyLoss, CombinedWeightedLoss, MixedLoss, WeightedDiceLoss, TverskyLoss, FocalLoss
from pseudo_label_generator import update_pseudo_labels


logging.basicConfig(
    format='%(asctime)s [%(levelname)s]     %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch U2CrackNet Segmentation Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default=None, help='Device: gpu or cpu')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--epochs', type=int, default=None, help='Total epochs')
    parser.add_argument('--save_interval', type=int, default=None, help='Save interval (in epochs)')
    parser.add_argument('--log_iters', type=int, default=None, help='Log interval (in iterations)')
    parser.add_argument('--save_dir', type=str, default=None, help='Save directory')
    parser.add_argument('--do_eval', action='store_true', help='Do evaluation')
    parser.add_argument('--use_vdl', action='store_true', help='Use VisualDL')
    
    
    parser.add_argument('--update_hard_thresh', type=float, default=None)
    parser.add_argument('--update_interval', type=int, default=None, help='Update interval (in epochs)')
    parser.add_argument('--update_start_ratio', type=float, default=None)
    parser.add_argument('--update_ratio_step', type=float, default=None)
    parser.add_argument('--update_max_ratio', type=float, default=None)
    parser.add_argument('--denoise_top_ratio', type=float, default=None)
    parser.add_argument('--denoise_trigger_unlock_ratio', type=float, default=None)
    
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
                
                if value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            except:
                pass
            d[keys[-1]] = value

    
    def resolve(arg_name, config_key, default_val):
        
        if getattr(args, arg_name) is not None:
            return getattr(args, arg_name)
        
        if config_key in config:
            return config[config_key]
        
        return default_val

    
    config['batch_size'] = resolve('batch_size', 'batch_size', 16)
    config['num_workers'] = resolve('num_workers', 'num_workers', 4)
    config['epochs'] = resolve('epochs', 'epochs', 1000)
    args.epochs = config['epochs'] 
    
    config['save_interval'] = resolve('save_interval', 'save_interval', 5)
    args.save_interval = config['save_interval'] 
    
    config['log_iters'] = resolve('log_iters', 'log_iters', 10)
    
    config['update_interval'] = resolve('update_interval', 'update_interval', 15)
    args.update_interval = config['update_interval'] 

    
    args.update_hard_thresh = resolve('update_hard_thresh', 'update_hard_thresh', 0.9)
    args.update_start_ratio = resolve('update_start_ratio', 'update_start_ratio', 0.1)
    args.update_ratio_step = resolve('update_ratio_step', 'update_ratio_step', 0.15)
    args.update_max_ratio = resolve('update_max_ratio', 'update_max_ratio', 0.85)
    args.denoise_top_ratio = resolve('denoise_top_ratio', 'denoise_top_ratio', 0.1)
    args.denoise_trigger_unlock_ratio = resolve('denoise_trigger_unlock_ratio', 'denoise_trigger_unlock_ratio', 0.01)

    
    lr = args.learning_rate
    if lr is None:
        if 'lr_scheduler' in config and 'learning_rate' in config['lr_scheduler']:
            lr = config['lr_scheduler']['learning_rate']
        else:
            lr = 0.001
    if 'lr_scheduler' not in config:
        config['lr_scheduler'] = {}
    config['lr_scheduler']['learning_rate'] = lr

    
    if args.device is None:
        args.device = 'gpu'
        
    
    if args.save_dir is None:
        args.save_dir = './output'
    
    return config


def check_multi_level_labels(dataset):
    """Detect mask type by reading raw label files when possible.

    Returns (has_multi_level, unique_values_set)
    """
    unique_values = set()
    try:
        
        iterable = None
        if hasattr(dataset, 'img_files') and isinstance(dataset.img_files, (list, tuple)):
            iterable = dataset.img_files
        elif hasattr(dataset, 'file_list') and isinstance(dataset.file_list, (list, tuple)):
            iterable = dataset.file_list
            
        if iterable is not None and len(iterable) > 0:
            sample_count = min(100, len(iterable))
            from PIL import Image
            for i in range(sample_count):
                try:
                    entry = iterable[i]
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        _, label_path = entry[0], entry[1]
                        if not os.path.isabs(label_path) and hasattr(dataset, 'data_root'):
                            label_path = os.path.join(dataset.data_root, label_path)
                    else:
                        continue

                    if os.path.exists(label_path):
                        with Image.open(label_path) as img:
                            arr = np.array(img)
                        unique_values.update(np.unique(arr).tolist())
                except Exception:
                    continue
        else:
            
            sample_count = min(100, len(dataset))
            for i in range(sample_count):
                try:
                    sample = dataset[i]
                    label = sample.get('label', None) if isinstance(sample, dict) else None
                    if label is None:
                        continue
                    if isinstance(label, torch.Tensor):
                        label = label.numpy()
                    unique_values.update(np.unique(label).tolist())
                except Exception:
                    continue
    except Exception:
        pass

    has_51 = 51 in unique_values
    has_204 = 204 in unique_values
    return (has_51 or has_204), unique_values


def calculate_pixel_statistics(dataset, dataset_name="训练集", save_dir=None, round_id=None):
    """Calculate pixel distribution by reading raw label files when possible.

    This mirrors PaddleSeg's `count_mask_pixels` behavior to avoid transform effects.
    """
    
    # logger.info(f"开始统计{dataset_name}的像素分布...")
    
    total_pixels = 0
    background_pixels = 0
    uncertain_pixels = 0
    foreground_pixels = 0
    
    
    very_low_pixels = 0  
    high_pixels = 0      
    extremely_high_pixels = 0  
    
    
    is_four_value_mask = False
    sample_count = 0
    
    try:
        from PIL import Image

        
        
        if hasattr(dataset, 'img_files') and getattr(dataset, 'img_files') is not None:
            iterable = dataset.img_files
        elif hasattr(dataset, 'file_list') and isinstance(dataset.file_list, (list, tuple)):
            iterable = dataset.file_list
        else:
            iterable = None

        if iterable is not None:
            for i in range(len(iterable)):
                try:
                    entry = iterable[i]
                    
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        _, label_path = entry[0], entry[1]
                    else:
                        
                        parts = entry if isinstance(entry, (list, tuple)) else str(entry).split()
                        if len(parts) >= 2:
                            label_path = parts[1]
                        elif len(parts) == 1:
                            
                            img_rel = parts[0]
                            label_path = img_rel.replace('images', 'labels').rsplit('.', 1)[0] + '.png'
                        else:
                            continue

                    
                    if not os.path.isabs(label_path) and hasattr(dataset, 'data_root'):
                        label_path_abs = os.path.join(dataset.data_root, label_path)
                    else:
                        label_path_abs = label_path

                    if os.path.exists(label_path_abs):
                        with Image.open(label_path_abs) as img:
                            mask_array = np.array(img)

                        
                        if sample_count == 0:
                            unique_vals = np.unique(mask_array)
                            is_four_value_mask = any(v in unique_vals for v in [51, 204])

                        total_pixels += mask_array.size
                        sample_count += 1

                        if is_four_value_mask:
                            
                            background_pixels += np.sum(mask_array == 0)
                            very_low_pixels += np.sum(mask_array == 51)  
                            high_pixels += np.sum(mask_array == 204)
                            extremely_high_pixels += np.sum(mask_array == 255)
                        else:
                            
                            background_pixels += np.sum(mask_array == 0)
                            uncertain_pixels += np.sum(mask_array == 128)
                            foreground_pixels += np.sum(mask_array == 255)
                    else:
                        
                        continue
                except Exception as e:
                    print(f"处理样本{i}时出错: {e}")
                    continue
    except Exception as e:
        logger.error(f"访问数据集时出错: {e}")
        pass

    if total_pixels > 0:
        if is_four_value_mask:
            
            foreground_pixels = very_low_pixels + high_pixels + extremely_high_pixels
            uncertain_pixels = very_low_pixels  
            
            bg_ratio = background_pixels/total_pixels*100
            uncertain_ratio = uncertain_pixels/total_pixels*100
            fg_ratio = foreground_pixels/total_pixels*100
            train_ratio = (background_pixels + high_pixels + extremely_high_pixels)/total_pixels*100  
            
            very_low_ratio = very_low_pixels/total_pixels*100
            high_ratio = high_pixels/total_pixels*100
            extremely_high_ratio = extremely_high_pixels/total_pixels*100
            
            # logger.info(f"{dataset_name}像素统计结果（四值掩码）:")
            # logger.info(f"  总像素数: {total_pixels:,}")
            # logger.info(f"  背景像素 (0): {background_pixels:,} ({bg_ratio:.2f}%)")
            # logger.info(f"  高置信前景 (255): {extremely_high_pixels:,} ({extremely_high_ratio:.2f}%)")
            # logger.info(f"  中置信前景 (204): {high_pixels:,} ({high_ratio:.2f}%)")
            # logger.info(f"  极低置信前景 (51) [不确定区域]: {very_low_pixels:,} ({very_low_ratio:.2f}%)")
            # logger.info(f"  所有前景区域总和: {foreground_pixels:,} ({fg_ratio:.2f}%)")
            # logger.info(f"  参与训练的像素: {background_pixels + high_pixels + extremely_high_pixels:,} ({train_ratio:.2f}%)")
            # logger.info(f"  不参与训练的像素（不确定区域）: {uncertain_pixels:,} ({uncertain_ratio:.2f}%)")
        else:
            
            bg_ratio = background_pixels/total_pixels*100
            uncertain_ratio = uncertain_pixels/total_pixels*100
            fg_ratio = foreground_pixels/total_pixels*100
            train_ratio = (background_pixels + foreground_pixels)/total_pixels*100
            
            # logger.info(f"{dataset_name}像素统计结果（三值掩码）:")
            # logger.info(f"  总像素数: {total_pixels:,}")
            # logger.info(f"  背景像素 (0): {background_pixels:,} ({bg_ratio:.2f}%)")
            # logger.info(f"  不确定像素 (128): {uncertain_pixels:,} ({uncertain_ratio:.2f}%)")
            # logger.info(f"  前景像素 (255): {foreground_pixels:,} ({fg_ratio:.2f}%)")
            # logger.info(f"  参与训练的像素: {background_pixels + foreground_pixels:,} ({train_ratio:.2f}%)")
            # logger.info(f"  不参与训练的像素: {uncertain_pixels:,} ({uncertain_ratio:.2f}%)")
        
        
        # if save_dir:
        #     from datetime import datetime
        #     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            
        #     os.makedirs(save_dir, exist_ok=True)
            
        #     stats_file = os.path.join(save_dir, "pixel_statistics.txt")
        #     with open(stats_file, "a", encoding="utf-8") as f:
        #         f.write(f"\n=== {dataset_name}像素统计结果 ===\n")
        #         f.write(f"统计时间: {timestamp}\n")
        #         if round_id is not None:
        #             f.write(f"训练轮次: Round {round_id}\n")
        #         f.write(f"总像素数: {total_pixels:,}\n")
                
        #         if is_four_value_mask:
        #             f.write(f"掩码类型: 四值掩码\n")
        #             f.write(f"背景像素 (0): {background_pixels:,} ({bg_ratio:.2f}%)\n")
        #             f.write(f"高置信前景 (255): {extremely_high_pixels:,} ({extremely_high_ratio:.2f}%)\n")
        #             f.write(f"中置信前景 (204): {high_pixels:,} ({high_ratio:.2f}%)\n")
        #             f.write(f"极低置信前景 (51) [不确定区域]: {very_low_pixels:,} ({very_low_ratio:.2f}%)\n")
        #             f.write(f"所有前景区域总和: {foreground_pixels:,} ({fg_ratio:.2f}%)\n")
        #             f.write(f"参与训练的像素: {background_pixels + high_pixels + extremely_high_pixels:,} ({train_ratio:.2f}%)\n")
        #             f.write(f"不参与训练的像素（不确定区域）: {uncertain_pixels:,} ({uncertain_ratio:.2f}%)\n")
        #         else:
        #             f.write(f"掩码类型: 三值掩码\n")
        #             f.write(f"背景像素 (0): {background_pixels:,} ({bg_ratio:.2f}%)\n")
        #             f.write(f"不确定像素 (128): {uncertain_pixels:,} ({uncertain_ratio:.2f}%)\n")
        #             f.write(f"前景像素 (255): {foreground_pixels:,} ({fg_ratio:.2f}%)\n")
        #             f.write(f"参与训练的像素: {background_pixels + foreground_pixels:,} ({train_ratio:.2f}%)\n")
        #             f.write(f"不参与训练的像素: {uncertain_pixels:,} ({uncertain_ratio:.2f}%)\n")
                
        #         f.write("=" * 50 + "\n")
        
        
        if is_four_value_mask:
            return {
                'total_pixels': total_pixels,
                'background_pixels': background_pixels,
                'uncertain_pixels': uncertain_pixels,  
                'foreground_pixels': foreground_pixels,  
                'very_low_pixels': very_low_pixels,
                'high_pixels': high_pixels,
                'extremely_high_pixels': extremely_high_pixels,
                'bg_ratio': bg_ratio,
                'uncertain_ratio': uncertain_ratio,
                'fg_ratio': fg_ratio,
                'train_ratio': train_ratio
            }
        else:
            return {
                'total_pixels': total_pixels,
                'background_pixels': background_pixels,
                'uncertain_pixels': uncertain_pixels,
                'foreground_pixels': foreground_pixels,
                'bg_ratio': bg_ratio,
                'uncertain_ratio': uncertain_ratio,
                'fg_ratio': fg_ratio,
                'train_ratio': train_ratio
            }
    else:
        print(f"无法统计{dataset_name}的像素分布")
        return {
            'total_pixels': 0,
            'background_pixels': 0,
            'uncertain_pixels': 0,
            'foreground_pixels': 0,
            'bg_ratio': 0,
            'uncertain_ratio': 0,
            'fg_ratio': 0,
            'train_ratio': 0
        }


def prune_dirs_by_patterns(base_dir, patterns, keep=3):
    """Keep only the latest `keep` directories matching each glob pattern under base_dir.

    Sorting preference: numeric suffix in directory name (e.g., 'prob_round2' -> 2). If no numeric
    suffix, fall back to modification time.
    """
    removed = []
    for pat in patterns:
        full_pat = os.path.join(base_dir, pat)
        matches = [p for p in glob.glob(full_pat) if os.path.isdir(p)]
        if not matches:
            continue

        def key_fn(p):
            bn = os.path.basename(p)
            m = re.search(r"(\d+)$", bn)
            if m:
                return int(m.group(1))
            try:
                return int(os.path.getmtime(p))
            except Exception:
                return 0

        matches_sorted = sorted(matches, key=key_fn)
        to_delete = matches_sorted[:-keep] if len(matches_sorted) > keep else []
        for d in to_delete:
            try:
                shutil.rmtree(d)
                # logger.info(f'[CLEANUP] 删除旧目录: {d}')
                removed.append(d)
            except Exception as e:
                # logger.warning(f'[CLEANUP] 删除目录失败: {d} -> {e}')
                pass
    return removed


def prune_checkpoints(save_dir, keep=5):
    """Keep only the latest `keep` checkpoint files named `model_iter_*.pth` in `save_dir`."""
    pattern = os.path.join(str(save_dir), 'model_iter_*.pth')
    files = [f for f in glob.glob(pattern) if os.path.isfile(f)]
    if not files:
        return []

    def key_fn(f):
        bn = os.path.basename(f)
        m = re.search(r'model_iter_(\d+)\.pth', bn)
        if m:
            return int(m.group(1))
        try:
            return int(os.path.getmtime(f))
        except Exception:
            return 0

    files_sorted = sorted(files, key=key_fn)
    to_delete = files_sorted[:-keep] if len(files_sorted) > keep else []
    removed = []
    for f in to_delete:
        try:
            os.remove(f)
            # logger.info(f'[CLEANUP] 删除旧检查点: {f}')
            removed.append(f)
        except Exception as e:
            # logger.warning(f'[CLEANUP] 删除检查点失败: {f} -> {e}')
            pass
    return removed


def cleanup_old_rounds(save_dir: str, data_root: str, current_round_id: int, keep_count: int = 3):
    if keep_count < 1:
        keep_count = 1
    all_round_ids = set()
    if os.path.isdir(save_dir):
        for item in os.listdir(save_dir):
            m = re.match(r'(?:prob_round|best_recall_round)(\d+)$', item)
            if m:
                try:
                    all_round_ids.add(int(m.group(1)))
                except Exception:
                    pass
    if data_root and os.path.isdir(data_root):
        for item in os.listdir(data_root):
            m = re.match(r'labels(\d+)(?:_pred)?$', item)
            if m:
                try:
                    all_round_ids.add(int(m.group(1)))
                except Exception:
                    pass
            m = re.match(r'train_round(\d+)\.txt$', item)
            if m:
                try:
                    all_round_ids.add(int(m.group(1)))
                except Exception:
                    pass
    if not all_round_ids:
        return []
    keep_round_ids = set()
    keep_round_ids.add(current_round_id)
    if current_round_id > 0:
        keep_round_ids.add(current_round_id - 1)
    sorted_round_ids = sorted(all_round_ids)
    if len(keep_round_ids) < keep_count:
        additional = [r for r in sorted_round_ids if r not in keep_round_ids]
        need = keep_count - len(keep_round_ids)
        if need > 0:
            keep_round_ids.update(additional[-need:])
    delete_round_ids = [rid for rid in sorted_round_ids if rid not in keep_round_ids]
    removed = []
    for rid in delete_round_ids:
        prob_dir = os.path.join(save_dir, f'prob_round{rid}')
        if os.path.isdir(prob_dir):
            try:
                shutil.rmtree(prob_dir)
                # logger.info(f'[CLEANUP] 删除旧目录: {prob_dir}')
                removed.append(prob_dir)
            except Exception as e:
                # logger.warning(f'[CLEANUP] 删除目录失败: {prob_dir} -> {e}')
                pass
        best_dir = os.path.join(save_dir, f'best_recall_round{rid}')
        if os.path.isdir(best_dir):
            try:
                shutil.rmtree(best_dir)
                # logger.info(f'[CLEANUP] 删除旧目录: {best_dir}')
                removed.append(best_dir)
            except Exception as e:
                # logger.warning(f'[CLEANUP] 删除目录失败: {best_dir} -> {e}')
                pass
        if data_root and os.path.isdir(data_root):
            labels_dir = os.path.join(data_root, f'labels{rid}')
            if os.path.isdir(labels_dir):
                try:
                    shutil.rmtree(labels_dir)
                    # logger.info(f'[CLEANUP] 删除旧目录: {labels_dir}')
                    removed.append(labels_dir)
                except Exception as e:
                    # logger.warning(f'[CLEANUP] 删除目录失败: {labels_dir} -> {e}')
                    pass
            labels_pred_dir = os.path.join(data_root, f'labels{rid}_pred')
            if os.path.isdir(labels_pred_dir):
                try:
                    shutil.rmtree(labels_pred_dir)
                    # logger.info(f'[CLEANUP] 删除旧目录: {labels_pred_dir}')
                    removed.append(labels_pred_dir)
                except Exception as e:
                    # logger.warning(f'[CLEANUP] 删除目录失败: {labels_pred_dir} -> {e}')
                    pass
            train_list_file = os.path.join(data_root, f'train_round{rid}.txt')
            if os.path.isfile(train_list_file):
                try:
                    os.remove(train_list_file)
                    # logger.info(f'[CLEANUP] 删除旧文件: {train_list_file}')
                    removed.append(train_list_file)
                except Exception as e:
                    # logger.warning(f'[CLEANUP] 删除文件失败: {train_list_file} -> {e}')
                    pass
    return removed


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)


def create_dataloader(config, mode='train', args=None, use_six_value=False, train_list_path=None):
    """Create data loader.
    
    Args:
        config: Configuration dict
        mode: 'train' or 'val'
        args: Arguments namespace
        use_six_value: Whether to use six-value label conversion
        train_list_path: Optional override path for training list (for pseudo-label updates)
    """
    dataset_cfg = config[f'{mode}_dataset']
    
    
    transforms_cfg = dataset_cfg.get('transforms', [])
    transforms = []
    
    for t_cfg in transforms_cfg:
        t_type = t_cfg.get('type')
        if t_type == 'Resize':
            transforms.append(Resize(t_cfg.get('size', [256, 256])))
        elif t_type == 'RandomPaddingCrop':
            transforms.append(RandomCrop(t_cfg.get('crop_size', [256, 256])))
        elif t_type == 'RandomCrop':
            transforms.append(RandomCrop(t_cfg.get('crop_size', [256, 256])))
        elif t_type == 'RandomHorizontalFlip':
            transforms.append(RandomHorizontalFlip(prob=0.5))
        elif t_type == 'Normalize':
            transforms.append(Normalize())
        elif t_type == 'FourValueLabelConvert' or t_type == 'SixValueLabelConvert':
            transforms.append(FourValueLabelConvert(
                extremely_high_weight=t_cfg.get('extremely_high_weight', 1.5),
                high_weight=t_cfg.get('high_weight', 0.5),
                background_weight=t_cfg.get('background_weight', 1.0),
                ignore_index=t_cfg.get('ignore_index', 255),
                ce_weights=t_cfg.get('ce_weights', None),
                dice_weights=t_cfg.get('dice_weights', None)
            ))
        elif t_type == 'BinaryLabelConvert':
            transforms.append(BinaryLabelConvert())
    
    
    if not any([type(t).__name__ in ('FourValueLabelConvert', 'SixValueLabelConvert', 'BinaryLabelConvert') for t in transforms]):
        if use_six_value:
            transforms.append(FourValueLabelConvert())
        else:
            transforms.append(BinaryLabelConvert())
    
    transform_compose = Compose(transforms)
    
    
    file_list_path = train_list_path if (mode == 'train' and train_list_path) else dataset_cfg[f'{mode}_path']
    
    dataset = CrackDataset(
        data_root=dataset_cfg['dataset_root'],
        file_list=file_list_path,
        mode=mode,
        num_classes=dataset_cfg.get('num_classes', 2),
        transforms=[transform_compose]
    )
    
    if mode == 'train':
        batch_size = config.get('batch_size', 16)
    else:
        batch_size = 1

    num_workers = config.get('num_workers', 4)
    
    
    g = None
    if 'seed' in config:
        g = torch.Generator()
        g.manual_seed(int(config['seed']))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        collate_fn=get_collate_fn(),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    return dataloader, dataset


def build_model(config, device):
    """Build model."""
    model_cfg = config.get('model', {})
    model_type = model_cfg.get('type', 'U2CrackNetImp19')
    num_classes = model_cfg.get('num_classes', 2)
    
    
    pretrained = model_cfg.get('pretrained', None)
    model = get_model(model_type, num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model


def build_optimizer(config, model):
    """Build optimizer."""
    opt_cfg = config.get('optimizer', {})
    opt_type = opt_cfg.get('type', 'adam')
    lr = config.get('lr_scheduler', {}).get('learning_rate', 0.001)
    
    if opt_type.lower() == 'adam':
        weight_decay = opt_cfg.get('weight_decay', 1e-4)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type.lower() == 'adamw':
        weight_decay = opt_cfg.get('weight_decay', 1e-4)
        
        
        beta1 = opt_cfg.get('momentum', 0.9)
        beta2 = opt_cfg.get('beta2', 0.999)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    elif opt_type.lower() == 'sgd':
        weight_decay = opt_cfg.get('weight_decay', 1e-4)
        momentum = opt_cfg.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    return optimizer


def build_lr_scheduler(optimizer, config, total_iters):
    """Build learning rate scheduler."""
    scheduler_cfg = config.get('lr_scheduler', {})
    
    
    
    
    power = scheduler_cfg.get('power', 0.9)
    warmup_iters = scheduler_cfg.get('warmup_iters', 0)
    warmup_factor = scheduler_cfg.get('warmup_factor', 0.001)
    
    def lr_lambda(current_step):
        if current_step < warmup_iters:
            
            alpha = warmup_factor
            return alpha + (1 - alpha) * (float(current_step) / float(warmup_iters))
        else:
            
            decay_steps = total_iters - warmup_iters
            if decay_steps <= 0:
                return 0.0
            
            step = current_step - warmup_iters
            step = min(step, decay_steps)
            return (1 - float(step) / float(decay_steps)) ** power

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_loss(config):
    """Build loss function."""
    loss_cfg = config.get('loss', {})
    loss_types = loss_cfg.get('types', [])
    coef = loss_cfg.get('coef', [])
    
    if not loss_types:
        
        
        raise ValueError("Loss configuration is required but not found in config")
    
    
    if len(loss_types) > 1:
        losses = []
        for loss_config in loss_types:
            loss_type = loss_config.get('type', 'CrossEntropyLoss')
            if loss_type in ['CrossEntropyLoss', 'WeightedCrossEntropyLoss']:
                ce_weight = loss_config.get('weight', [1.0, 1.0])
                ignore_index = loss_config.get('ignore_index', 255)
                
                losses.append(WeightedCrossEntropyLoss(
                    weight=ce_weight,
                    ignore_index=ignore_index,
                    channel_index=0
                ))
            elif loss_type in ['DiceLoss', 'WeightedDiceLoss']:
                ignore_index = loss_config.get('ignore_index', 255)
                
                losses.append(WeightedDiceLoss(
                    ignore_index=ignore_index,
                    channel_index=1
                ))
            elif loss_type == 'CombinedWeightedLoss':
                losses.append(CombinedWeightedLoss(
                    ce_weight=loss_config.get('ce_weight', [1.0, 1.0]),
                    ignore_index=loss_config.get('ignore_index', 255),
                    pos_margin=loss_config.get('pos_margin', 0.5),
                    neg_margin=loss_config.get('neg_margin', 0.3),
                    edge_loss_weight=loss_config.get('edge_loss_weight', 0.5)
                ))
            elif loss_type == 'FocalLoss':
                ignore_index = loss_config.get('ignore_index', 255)
                
                channel_idx = loss_config.get('channel_index', 0)
                losses.append(FocalLoss(
                    alpha=loss_config.get('alpha', 0.25),
                    gamma=loss_config.get('gamma', 2.0),
                    ignore_index=ignore_index,
                    channel_index=channel_idx
                ))
            elif loss_type in ['TverskyLoss', 'WeightedTverskyLoss']:
                ignore_index = loss_config.get('ignore_index', 255)
                
                
                
                
                channel_idx = loss_config.get('channel_index', 1 if 'CrossEntropyLoss' in [l.get('type') for l in loss_types] else 0)
                losses.append(TverskyLoss(
                    alpha=loss_config.get('alpha', 0.7),
                    beta=loss_config.get('beta', 0.3),
                    ignore_index=ignore_index,
                    channel_index=channel_idx
                ))
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")
        
        if not coef:
            coef = [1.0] * len(losses)
        
        return MixedLoss(losses, coef)
    
    loss_config = loss_types[0]
    loss_type = loss_config.get('type', 'CrossEntropyLoss')
    
    if loss_type in ['CrossEntropyLoss', 'WeightedCrossEntropyLoss']:
        
        ce_weight = loss_config.get('weight', [1.0, 1.0])  
        ignore_index = loss_config.get('ignore_index', 255)
        return WeightedCrossEntropyLoss(
            weight=ce_weight,
            ignore_index=ignore_index
        )
    elif loss_type == 'CombinedWeightedLoss':
        return CombinedWeightedLoss(
            ce_weight=loss_config.get('ce_weight', [1.0, 1.0]),
            ignore_index=loss_config.get('ignore_index', 255),
            pos_margin=loss_config.get('pos_margin', 0.5),
            neg_margin=loss_config.get('neg_margin', 0.3),
            edge_loss_weight=loss_config.get('edge_loss_weight', 0.5)
        )
    elif loss_type == 'FocalLoss':
        return FocalLoss(
            alpha=loss_config.get('alpha', 0.25),
            gamma=loss_config.get('gamma', 2.0),
            ignore_index=loss_config.get('ignore_index', 255),
            channel_index=loss_config.get('channel_index', 0)
        )
    elif loss_type in ['TverskyLoss', 'WeightedTverskyLoss']:
        return TverskyLoss(
            alpha=loss_config.get('alpha', 0.7),
            beta=loss_config.get('beta', 0.3),
            ignore_index=loss_config.get('ignore_index', 255),
            channel_index=loss_config.get('channel_index', 0)
        )
    elif loss_type in ['DiceLoss', 'WeightedDiceLoss']:
        
        return WeightedDiceLoss(
            ignore_index=loss_config.get('ignore_index', 255),
            channel_index=loss_config.get('channel_index', 0)
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch, args):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        
        semantic_weights = batch.get('semantic_weights', None)
        if semantic_weights is not None:
            semantic_weights = semantic_weights.to(device)
            
        loss = criterion(outputs, labels, semantic_weights)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        if batch_idx % args.log_iters == 0:
            avg_loss = total_loss / total_samples
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {avg_loss:.6f}, LR: {lr:.6e}')
    
    avg_epoch_loss = total_loss / total_samples
    return avg_epoch_loss


def calculate_metrics(preds, labels, ignore_index=255):
    """Calculate IoU, Accuracy, Precision, Recall for each class."""
    num_classes = 2  
    
    
    preds = preds.flatten()
    labels = labels.flatten()
    
    
    valid_mask = labels != ignore_index
    preds = preds[valid_mask]
    labels = labels[valid_mask]
    
    
    class_ious = []
    class_precisions = []
    class_recalls = []
    
    total_correct = 0
    total_samples = len(labels)
    
    for class_id in range(num_classes):
        pred_mask = (preds == class_id)
        label_mask = (labels == class_id)
        
        
        tp = (pred_mask & label_mask).sum().item()
        fp = (pred_mask & ~label_mask).sum().item()
        fn = (~pred_mask & label_mask).sum().item()
        
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        class_ious.append(iou)
        
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        class_precisions.append(precision)
        
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        class_recalls.append(recall)
        
        total_correct += tp
    
    
    mIoU = np.mean(class_ious)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    
    po = accuracy  
    pe = 0  
    for class_id in range(num_classes):
        pred_prob = (preds == class_id).sum().item() / total_samples
        label_prob = (labels == class_id).sum().item() / total_samples
        pe += pred_prob * label_prob
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    
    
    
    fg_iou = class_ious[1] if num_classes > 1 else 0.0
    dice = 2 * fg_iou / (1 + fg_iou) if (1 + fg_iou) > 0 else 0.0
    
    return {
        'mIoU': mIoU,
        'accuracy': accuracy,
        'kappa': kappa,
        'dice': dice,
        'class_ious': class_ious,
        'class_precisions': class_precisions,
        'class_recalls': class_recalls
    }


def validate(model, dataloader, device, dataset_size=None):
    """Validate model and calculate metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    total_time = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            start_time = time.time()
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            total_time += time.time() - start_time

            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    
    metrics = calculate_metrics(torch.tensor(all_preds), torch.tensor(all_labels))
    
    
    fps = len(dataloader) / total_time if total_time > 0 else 0.0
    
    return metrics, fps


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
    
    

    
    



def create_train_eval_loader(config, file_list, args):
    """Create a deterministic dataloader for training set evaluation."""
    dataset_cfg = config['train_dataset']
    transforms_cfg = dataset_cfg.get('transforms', [])
    det_transforms = []
    
    for t_cfg in transforms_cfg:
        t_type = t_cfg.get('type')
        if t_type and t_type.startswith('Random'):
            continue
        
        if 'LabelConvert' in t_type:
            continue
        
        if t_type == 'Resize':
            det_transforms.append(Resize(t_cfg.get('size', [256, 256])))
        elif t_type == 'Normalize':
            det_transforms.append(Normalize())
            
    det_compose = Compose(det_transforms)
    
    dataset = CrackDataset(
        data_root=dataset_cfg['dataset_root'],
        file_list=file_list,
        mode='val', 
        num_classes=dataset_cfg.get('num_classes', 2),
        transforms=[det_compose]
    )
    
    
    
    batch_size = config.get('batch_size', 16)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=get_collate_fn(),
        pin_memory=True
    )
    
    return dataloader


def validate_on_train(model, dataloader, device):
    """Validate on training set with specific 4-value mask conversion."""
    model.eval()
    all_preds = []
    all_labels = []
    
    
    pbar = tqdm(dataloader, desc="Eval on Train", ncols=100, leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label']
            
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
                
            all_preds.append(preds)
            all_labels.append(labels)
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    
    
    mapped_labels = np.full_like(all_labels, 255, dtype=np.uint8)
    mapped_labels[all_labels == 0] = 0
    mapped_labels[all_labels == 204] = 1
    mapped_labels[all_labels == 255] = 1
    
    metrics = calculate_metrics(torch.tensor(all_preds), torch.tensor(mapped_labels), ignore_index=255)
    return metrics


def main():
    args = parse_args()
    
    
    config = load_config(args.config, args)
    
    
    if 'seed' in config:
        seed = int(config['seed'])
        logger.info(f'Setting random seed to {seed}')
        set_seed(seed)
    
    
    device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    
    logger.info('Loading datasets...')
    train_loader, train_dataset = create_dataloader(config, mode='train', args=args, use_six_value=False)
    
    
    logger.info('')
    
    
    dataset_cfg = config['train_dataset']
    temp_dataset = CrackDataset(
        data_root=dataset_cfg['dataset_root'],
        file_list=dataset_cfg['train_path'],
        mode='train',
        num_classes=dataset_cfg.get('num_classes', 2),
        transforms=[]  
    )
    
    has_multi_level, unique_vals = check_multi_level_labels(temp_dataset)

    if has_multi_level:
        # logger.info('=' * 50)
        # logger.info('检测到四值掩码（包含51或204）')
        # logger.info('将使用四值掩码更新策略')
        # logger.info('=' * 50)
        # logger.info('')
        pass

    
    try:
        pixel_stats = calculate_pixel_statistics(temp_dataset, "训练集", save_dir=str(save_dir), round_id=0)
    except Exception:
        pixel_stats = None
    # logger.info('')
    
    
    
    if 'seed' in config:
        logger.info(f'Re-setting random seed to {config["seed"]} before training initialization')
        set_seed(int(config['seed']))

    train_loader, train_dataset = create_dataloader(config, mode='train', args=args, use_six_value=has_multi_level)
    val_loader, val_dataset = create_dataloader(config, mode='val', args=args, use_six_value=has_multi_level)
    
    
    logger.info('Building model...')
    model = build_model(config, device)
    
    
    optimizer = build_optimizer(config, model)
    iters_per_epoch = len(train_loader)
    
    
    
    
    epochs = config.get('epochs', 1000)
    
    args.epochs = epochs
    
    total_iters = epochs * iters_per_epoch
    scheduler = build_lr_scheduler(optimizer, config, total_iters)
    
    
    
    save_interval_epochs = config.get('save_interval', 5)
    args.save_interval = save_interval_epochs 
    save_interval_iters = save_interval_epochs * iters_per_epoch
    
    
    logger.info('Building loss function...')
    criterion = build_loss(config)
    logger.info(f'Loss function built: {criterion}')
    criterion = criterion.to(device)
    
    
    logger.info('Starting training...')
    best_miou = 0.0
    best_miou_iter = 0
    iters = 0
    round_num = 0
    
    
    loss_history = []
    
    
    log_file = save_dir / "train_log.txt"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s]     %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger().addHandler(file_handler)

    
    # metrics_path = save_dir / "metrics.csv"
    # with open(metrics_path, "w", encoding="utf-8") as f:
    #     f.write("Epoch,Loss,LR\n")

    
    
    update_interval_epochs = config.get('update_interval', 15)
    args.update_interval = update_interval_epochs 
    update_interval = update_interval_epochs * iters_per_epoch
    
    def _list_train_images(train_list_path: str, data_root: str):
        pairs = []
        with open(train_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 1:
                    img_rel = parts[0]
                    lbl_rel = img_rel.replace('images', 'labels').rsplit('.', 1)[0] + '.png'
                else:
                    img_rel, lbl_rel = parts[0], parts[1]
                img_path = img_rel if os.path.isabs(img_rel) else os.path.join(dataset_cfg['dataset_root'], img_rel)
                lbl_path = lbl_rel if os.path.isabs(lbl_rel) else os.path.join(dataset_cfg['dataset_root'], lbl_rel)
                pairs.append((img_path, lbl_path))
        return pairs

    def _infer_and_save_bg_probs(model, cfg, train_list_abs: str, data_root_abs: str, out_dir: str, device: torch.device):
        os.makedirs(out_dir, exist_ok=True)
        
        ds_cfg = cfg['val_dataset'] if 'val_dataset' in cfg and cfg['val_dataset'] else cfg['train_dataset']
        
        transforms_cfg = ds_cfg.get('transforms', [])
        det_transforms = []
        for t in transforms_cfg:
            t_type = t.get('type')
            if t_type and t_type.startswith('Random'):
                continue
            det_transforms.append(t)
        
        det_list = []
        for t_cfg in det_transforms:
            t_type = t_cfg.get('type')
            if t_type == 'Resize':
                det_list.append(Resize(t_cfg.get('size', [448, 448])))
            elif t_type == 'Normalize':
                det_list.append(Normalize())
        det_compose = Compose(det_list)
        
        infer_ds = CrackDataset(
            data_root=data_root_abs,
            file_list=train_list_abs,
            mode='val',
            num_classes=ds_cfg.get('num_classes', 2),
            transforms=[det_compose]
        )
        infer_loader = DataLoader(
            infer_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=get_collate_fn(),
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(int(cfg.get('seed', 42))) if 'seed' in cfg else None
        )
        
        pairs = _list_train_images(train_list_abs, data_root_abs)
        base_names = [os.path.splitext(os.path.basename(p[0]))[0] for p in pairs]
        idx_counter = 0
        model.eval()
        with torch.no_grad():
            for batch in infer_loader:
                images = batch['image'].to(device)
                outputs = model(images)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                probs = torch.softmax(outputs, dim=1)
                bg_prob = probs[:, 0, :, :].squeeze(0).cpu().numpy().astype('float32')
                base = base_names[idx_counter] if idx_counter < len(base_names) else f"sample_{idx_counter:06d}"
                np.save(os.path.join(out_dir, base + '.npy'), bg_prob)
                idx_counter += 1
        return len(base_names)

    while iters < total_iters:
        
        recall_dir = save_dir / f'best_recall_round{round_num}'
        recall_dir.mkdir(parents=True, exist_ok=True)
        
        
        round_start = iters
        round_end = min(round_start + update_interval, total_iters)
        round_start_epoch = round_start // iters_per_epoch
        round_end_epoch = round_end // iters_per_epoch
        round_end_str = f"{round_start_epoch}-{round_end_epoch} epoch"
        
        # logger.info(f'\n开始第{round_num}段训练: {round_end_str}')
        # logger.info('使用自定义训练函数，包含标签转换逻辑和可视化功能...')
        # logger.info(f'为第{round_num}段训练创建独立的best_recall目录: {recall_dir}')
        
        if has_multi_level:
            
            sv = None
            try:
                
                comp = train_dataset.transforms[0]
                for t in getattr(comp, 'transforms', []):
                    if type(t).__name__ == 'FourValueLabelConvert':
                        sv = t
                        break
            except Exception:
                sv = None

            if sv is not None:
                # logger.info('FourValueLabelConvert: 检测到四值掩码')
                if getattr(sv, 'use_multi_channel', False):
                    # logger.info('  模式: 多通道权重 (Multi-channel Weights)')
                    # logger.info(f'  CE Weights (Channel 0): {sv.ce_weights}')
                    # logger.info(f'  Dice/Tversky Weights (Channel 1): {sv.dice_weights}')
                    pass
                else:
                    # logger.info(f'  极高置信(255)权重: {sv.extremely_high_weight}')
                    # logger.info(f'  高置信(204)权重: {sv.high_weight}')
                    # logger.info(f'  背景(0)权重: {sv.background_weight}')
                    pass
                # logger.info(f'  极低置信(51)区域: 不参与损失计算 (ignore_index={sv.ignore_index})')
            else:
                # logger.info('FourValueLabelConvert: 检测到四值掩码 (使用默认权重)')
                # logger.info('  极高置信(255)权重: 1.5')
                # logger.info('  高置信(204)权重: 0.5')
                # logger.info('  背景(0)权重: 1.0')
                # logger.info('  极低置信(51)区域: 不参与损失计算 (ignore_index=255)')
                pass
        
        best_recall_round = 0.0
        best_recall_iter_round = 0
        
        
        while iters < round_end and iters < total_iters:
            
            epoch_loss = 0.0
            num_batches = 0
            
            
            current_epoch = iters // iters_per_epoch + 1
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                       desc=f"Epoch {current_epoch}/{args.epochs}", ncols=100)
            
            for epoch_batch_idx, batch in pbar:
                if iters >= round_end or iters >= total_iters:
                    break
                
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                semantic_weights = batch.get('semantic_weights', None)
                if semantic_weights is not None:
                    semantic_weights = semantic_weights.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                if isinstance(outputs, (list, tuple)):
                    loss = 0
                    for o in outputs:
                        loss += criterion(o, labels, semantic_weights=semantic_weights)
                else:
                    loss = criterion(outputs, labels, semantic_weights=semantic_weights)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                
                loss_val = loss.item()
                epoch_loss += loss_val
                num_batches += 1
                
                iters += 1
                
                
                pbar.set_postfix({'loss': f'{loss_val:.4f}'})
                

            
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                lr = optimizer.param_groups[0]['lr']
                
                
                logger.info(f"[TRAIN] Epoch: {current_epoch}/{args.epochs}, Average Loss: {avg_loss:.4f}, LR: {lr:.6f}")
                
                
                # with open(metrics_path, "a", encoding="utf-8") as f:
                #     f.write(f"{current_epoch},{avg_loss:.4f},{lr:.6f}\n")
                
                
                loss_history.append(avg_loss)
                try:
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Train Loss', marker='o')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Training Loss per Epoch')
                    plt.grid(True)
                    plt.legend()
                    plt.savefig(save_dir / "loss_curve.png")
                    plt.close()
                except Exception as e:
                    logger.warning(f"Plotting failed: {e}")

            
            if (current_epoch % args.save_interval == 0 or iters == total_iters) and args.do_eval:
                total_samples = len(val_dataset)
                num_iters = len(val_loader)
                if total_samples == 0 or num_iters == 0:
                    # logger.info(f'跳过评估（验证集为空）：total_samples={total_samples}, total_iters={num_iters}')
                    ckpt_path = save_dir / f'model_iter_{iters}.pth'
                    torch.save(model.state_dict(), ckpt_path)
                    try:
                        prune_checkpoints(save_dir, keep=5)
                    except Exception as e:
                        # logger.warning(f'[CLEANUP] prune_checkpoints 失败: {e}')
                        pass
                else:
                    logger.info(f'Start evaluating (total_samples: {total_samples}, total_iters: {num_iters})...')
                    metrics, fps = validate(model, val_loader, device)
                    
                    logger.info(f'[EVAL] Image-level #Images: {total_samples} mIoU: {metrics["mIoU"]:.4f} Acc: {metrics["accuracy"]:.4f}')
                    logger.info(f'[EVAL] Image-level Class IoU:')
                    logger.info(str(np.array(metrics["class_ious"])))
                    logger.info(f'[EVAL] Image-level Class Precision:')
                    logger.info(str(np.array(metrics["class_precisions"])))
                    logger.info(f'[EVAL] Image-level Class Recall:')
                    logger.info(str(np.array(metrics["class_recalls"])))
                    ckpt_path = save_dir / f'model_iter_{iters}.pth'
                    torch.save(model.state_dict(), ckpt_path)
                    try:
                        prune_checkpoints(save_dir, keep=5)
                    except Exception as e:
                        # logger.warning(f'[CLEANUP] prune_checkpoints 失败: {e}')
                        pass
                    if metrics["mIoU"] > best_miou:
                        best_miou = metrics["mIoU"]
                        best_miou_iter = iters
                        # best_path = save_dir / 'best_model.pth'
                        # torch.save(model.state_dict(), best_path)
                    best_miou_epoch = (best_miou_iter - 1) // len(train_loader) + 1
                    # logger.info(f'[EVAL] The model with the best validation mIoU ({best_miou:.4f}) was saved at iter {best_miou_iter} (Epoch {best_miou_epoch}).')
                    
                    logger.info(f'Start evaluating on Training Set for Recall (Round {round_num})...')
                    try:
                        
                        train_eval_loader = create_train_eval_loader(config, train_dataset.file_list, args)
                        train_metrics = validate_on_train(model, train_eval_loader, device)
                        train_avg_recall = np.mean(train_metrics["class_recalls"]) if 'class_recalls' in train_metrics else 0.0
                        logger.info(f'[TRAIN-EVAL] Image-level Class Recall: {train_metrics.get("class_recalls", [])}')
                        logger.info(f'[TRAIN-EVAL] Average Recall: {train_avg_recall:.4f}')
                    except Exception as e:
                        logger.error(f"Error during training set evaluation: {e}")
                        train_avg_recall = 0.0

                    if train_avg_recall > best_recall_round:
                        best_recall_round = train_avg_recall
                        best_recall_iter_round = iters
                        best_recall_path = recall_dir / 'best_model.pth'
                        if not recall_dir.exists():
                            recall_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), best_recall_path)
                        best_recall_epoch = (iters - 1) // len(train_loader) + 1
                        logger.info(f'[EVAL] The model with the best TRAINING average recall ({best_recall_round:.4f}) was saved at iter {iters} (Epoch {best_recall_epoch}) in round {round_num}.')
                    else:
                        best_recall_epoch = (best_recall_iter_round - 1) // len(train_loader) + 1
                        logger.info(f'[EVAL] The model with the best TRAINING average recall ({best_recall_round:.4f}) was saved at iter {best_recall_iter_round} (Epoch {best_recall_epoch}) in round {round_num}.')

        
        
        if iters < total_iters:
            # logger.info('')
            # logger.info(f'使用第{round_num}段的best_recall模型进行动态更新: {recall_dir}')
            
            best_recall_model_path = recall_dir / 'best_model.pth'
            if best_recall_model_path.exists():
                # logger.info(f'加载best_recall模型: {best_recall_model_path}')
                model.load_state_dict(torch.load(best_recall_model_path, map_location=device))
            else:
                # logger.warning(f'未找到best_recall模型 ({best_recall_model_path})，将使用当前模型权重进行更新')
                pass
            
            
            data_root_abs = os.path.abspath(config['train_dataset']['dataset_root']).replace('\\', '/')
            train_list_abs = os.path.abspath(config['train_dataset']['train_path']).replace('\\', '/')
            
            prob_dir = os.path.join(str(save_dir), f'prob_round{round_num}')
            num_samples = _infer_and_save_bg_probs(model, config, train_list_abs, data_root_abs, prob_dir, device)
            # logger.info(f'已生成背景概率图: {num_samples} 个，保存于: {prob_dir}')
            
            upd_cmd = [sys.executable,
                       os.path.join('tools', 'update_tristate_labels.py'),
                       '--data_root', data_root_abs,
                       '--base_labels', os.path.join(data_root_abs, 'labels'),
                       '--prob_dir', prob_dir,
                       '--train_list', train_list_abs,
                       '--round_index', str(round_num),
                       '--hard_thresh', str(args.update_hard_thresh),
                       '--top_ratio', str(min(args.update_start_ratio + args.update_ratio_step * round_num, args.update_max_ratio)),
                       '--start_ratio', str(args.update_start_ratio),
                       '--ratio_step', str(args.update_ratio_step),
                       '--max_ratio', str(args.update_max_ratio),
                       '--six_unlock_cap', str(0.8),
                       '--denoise_top_ratio', str(args.denoise_top_ratio),
                       '--denoise_trigger_unlock_ratio', str(args.denoise_trigger_unlock_ratio)]
            result = subprocess.run(upd_cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                # logger.warning(f'[更新失败] 标签更新脚本返回码: {result.returncode}')
                # logger.warning(result.stderr)
                pass
            else:
                # logger.info('标签更新完成，切换训练列表...')
                
                new_train_list = os.path.join(data_root_abs, f'train_round{round_num}.txt')
                if os.path.isfile(new_train_list):
                    config['train_dataset']['train_path'] = new_train_list
                    
                    
                    temp_dataset = CrackDataset(
                        data_root=config['train_dataset']['dataset_root'],
                        file_list=new_train_list,
                        mode='train',
                        num_classes=config['train_dataset'].get('num_classes', 2),
                        transforms=[]
                    )
                    has_multi_level, _ = check_multi_level_labels(temp_dataset)
                    if has_multi_level:
                        # logger.info(f'Round {round_num} update: Detected multi-level labels, enabling FourValueLabelConvert.')
                        pass
                    
                    train_loader, train_dataset = create_dataloader(config, mode='train', args=args, use_six_value=has_multi_level)
                    # logger.info(f'训练数据集大小: {len(train_dataset)}')
                    
                    
                    _ = calculate_pixel_statistics(temp_dataset, f"训练集(Round {round_num})", save_dir=str(save_dir), round_id=round_num)

                    try:
                        cleanup_old_rounds(str(save_dir), data_root_abs, round_num, keep_count=3)
                    except Exception as e:
                        # logger.warning(f'[CLEANUP] 更新后清理旧目录失败: {e}')
                        pass
                else:
                    # logger.warning(f'[更新失败] 未找到新的训练列表: {new_train_list}')
                    pass
        
        round_num += 1
    
    logger.info('Training finished!')


if __name__ == '__main__':
    main()

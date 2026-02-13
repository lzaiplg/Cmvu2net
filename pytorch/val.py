import os
import sys
import argparse
import yaml
import time
import numpy as np
from pathlib import Path
import logging
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import get_model
from datasets import CrackDataset, Compose, Resize, Normalize, BinaryLabelConvert, FourValueLabelConvert, get_collate_fn


# Setup logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s]     %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Model Evaluation')

    # params of evaluate
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--opts',
        help='Update the key-value pairs of all options.',
        default=None,
        nargs='+')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for evaluation.',
        type=str,
        default=None)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Number of workers for data loader.',
        type=int,
        default=0)

    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='Directory to save predicted masks and overlay images.',
        type=str,
        default=None)

    parser.add_argument(
        '--device',
        dest='device',
        help='Device place to be set, which can be gpu, or cpu.',
        default='gpu',
        choices=['cpu', 'gpu'],
        type=str)
    
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Batch size for evaluation.',
        type=int,
        default=1)
    
    return parser.parse_args()


def calculate_metrics(preds, labels, ignore_index=255):
    num_classes = 2
    preds = preds.flatten()
    labels = labels.flatten()
    valid_mask = labels != ignore_index
    preds = preds[valid_mask]
    labels = labels[valid_mask]
    class_ious = []
    class_precisions = []
    class_recalls = []
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
    mIoU = np.mean(class_ious)
    accuracy = (preds == labels).sum().item() / total_samples if total_samples > 0 else 0.0
    kappa_val = 0.0
    dice_val = 0.0
    return {
        'mIoU': mIoU,
        'accuracy': accuracy,
        'kappa': kappa_val,
        'dice': dice_val,
        'class_ious': class_ious,
        'class_precisions': class_precisions,
        'class_recalls': class_recalls
    }


def calculate_area(pred, label, num_classes, ignore_index=255):
    """
    Calculate intersection, prediction and label areas for each class.
    """
    pred = pred.flatten()
    label = label.flatten()
    
    # Create mask for valid pixels (not ignore_index)
    mask = label != ignore_index
    pred = pred[mask]
    label = label[mask]
    
    # Calculate intersection
    intersect = pred[pred == label]
    intersect_area, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
    
    # Calculate prediction and label areas
    pred_area, _ = np.histogram(pred, bins=np.arange(num_classes + 1))
    label_area, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    
    return intersect_area, pred_area, label_area


def mean_iou(intersect_area, pred_area, label_area):
    """
    Calculate mean Intersection over Union (mIoU).
    """
    union_area = pred_area + label_area - intersect_area
    iou = intersect_area / (union_area + 1e-10)  # Add small value to avoid division by zero
    miou = np.mean(iou)
    return iou, miou


def class_measurement(intersect_area, pred_area, label_area):
    """
    Calculate accuracy, precision, and recall for each class.
    """
    # Overall accuracy
    total_correct = np.sum(intersect_area)
    total_pixels = np.sum(label_area)
    acc = total_correct / (total_pixels + 1e-10)
    
    # Per-class precision
    precision = intersect_area / (pred_area + 1e-10)
    
    # Per-class recall
    recall = intersect_area / (label_area + 1e-10)
    
    return acc, precision, recall


def kappa(intersect_area, pred_area, label_area):
    """
    Calculate kappa coefficient.
    """
    total = np.sum(label_area)
    Po = np.sum(intersect_area) / total
    Pe = np.sum(pred_area * label_area) / (total * total)
    kappa = (Po - Pe) / (1 - Pe + 1e-10)
    return kappa


def dice(intersect_area, pred_area, label_area):
    """
    Calculate Dice coefficient.
    """
    union_dice = pred_area + label_area
    dice_per_class = (2 * intersect_area) / (union_dice + 1e-10)
    mdice = np.mean(dice_per_class)
    return dice_per_class, mdice


def evaluate(model, eval_dataset, device, num_workers=0, save_dir=None, print_detail=True, batch_size=1):
    """
    Launch evaluation consistent with training metrics.

    Args:
        model: A semantic segmentation model.
        eval_dataset: Used to read and process validation datasets.
        device: Device to run the model on.
        num_workers: Number of workers for data loader.
        save_dir: Directory to save predicted masks and overlay images.
        print_detail: Whether to print detailed information about the evaluation process.
        batch_size: Batch size for evaluation.

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
        np.ndarray: Class IoU.
        np.ndarray: Class Precision.
        np.ndarray: Class Recall.
        float: Kappa coefficient.
    """
    model.eval()
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=get_collate_fn()
    )
    total_iters = len(loader)
    if print_detail:
        logger.info("Start evaluating (total_samples: {}, total_iters: {})...".format(len(eval_dataset), total_iters))
    eval_start_time = time.time()
    all_preds = []
    all_labels = []
    image_paths = []
    with torch.no_grad():
        for iter_idx, data in enumerate(loader):
            images = data['image'].to(device)
            labels = data['label'].numpy().astype('int64')
            batch_img_paths = data.get('image_path', [None] * len(images))
            preds = model(images)
            # Handle tuple output (e.g., from U2Net family) by taking the first element
            if isinstance(preds, tuple) or isinstance(preds, list):
                preds = preds[0]
            if isinstance(preds, dict):
                preds = preds['out'] if 'out' in preds else next(iter(preds.values()))
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            if preds.shape != labels.shape:
                resized_pred = []
                for i in range(preds.shape[0]):
                    resized_pred.append(
                        np.array(Image.fromarray(preds[i].astype('uint8')).resize(
                            (labels.shape[2], labels.shape[1]), resample=Image.NEAREST
                        ))
                    )
                preds = np.stack(resized_pred, axis=0)
            all_preds.append(preds)
            all_labels.append(labels)
            
            if save_dir is not None:
                os.makedirs(os.path.join(save_dir, 'masks'), exist_ok=True)
                os.makedirs(os.path.join(save_dir, 'overlay'), exist_ok=True)
                
                for i in range(len(preds)):
                    img_path = batch_img_paths[i]
                    if img_path:
                        img_name = os.path.basename(img_path)
                        img_name_without_ext = os.path.splitext(img_name)[0]
                        mask_name = f"{img_name_without_ext}.png"
                    else:
                        mask_name = f'{iter_idx}_{i}.png'
                    
                    mask_bin = (preds[i].astype(np.uint8) * 255)
                    if mask_bin.ndim == 3:
                        mask_bin = mask_bin.squeeze()
                    cv2.imwrite(os.path.join(save_dir, 'masks', mask_name), mask_bin)
                    
                    if img_path and os.path.exists(img_path):
                        orig_img = cv2.imread(img_path)
                        if orig_img is not None:
                            if mask_bin.shape != orig_img.shape[:2]:
                                mask_bin_resized = cv2.resize(mask_bin, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                            else:
                                mask_bin_resized = mask_bin
                            
                            overlay = orig_img.copy()
                            colored_mask = np.zeros_like(orig_img)
                            colored_mask[mask_bin_resized > 0] = [0, 0, 255]
                            overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
                            cv2.imwrite(os.path.join(save_dir, 'overlay', mask_name), overlay)
                    else:
                        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
                        img_np = ((img_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        
                        if mask_bin.shape != img_np.shape[:2]:
                            mask_bin_resized = cv2.resize(mask_bin, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                        else:
                            mask_bin_resized = mask_bin
                            
                        overlay = img_np.copy()
                        colored_mask = np.zeros_like(img_np)
                        colored_mask[mask_bin_resized > 0] = [0, 0, 255]
                        overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
                        cv2.imwrite(os.path.join(save_dir, 'overlay', mask_name), overlay)

    all_preds_np = np.concatenate(all_preds, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    metrics = calculate_metrics(torch.tensor(all_preds_np), torch.tensor(all_labels_np))
    if print_detail:
        logger.info("[EVAL] Image-level #Images: {} mIoU: {:.4f} Acc: {:.4f}".format(
            len(eval_dataset), metrics['mIoU'], metrics['accuracy']))
        logger.info("[EVAL] Image-level Class IoU: \n" + str(np.round(np.array(metrics['class_ious']), 8)))
        logger.info("[EVAL] Image-level Class Precision: \n" + str(np.round(np.array(metrics['class_precisions']), 8)))
        logger.info("[EVAL] Image-level Class Recall: \n" + str(np.round(np.array(metrics['class_recalls']), 8)))
    return metrics['mIoU'], metrics['accuracy'], np.array(metrics['class_ious']), np.array(metrics['class_precisions']), np.array(metrics['class_recalls']), metrics['kappa']


def load_config(config_path, args):
    """Load and merge config from YAML and command line."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Apply command line overrides
    if args.opts:
        for i in range(0, len(args.opts), 2):
            key, value = args.opts[i], args.opts[i+1]
            # Parse nested keys like "val_dataset.dataset_root"
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            # Convert value to appropriate type
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string
            d[keys[-1]] = value
    
    return config


def main(args):
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    config = load_config(args.cfg, args)
    
    # Check device
    if args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")

    # Create validation dataset
    val_dataset_config = config.get('val_dataset', {})
    if not val_dataset_config:
        raise RuntimeError('The validation dataset is not specified in the configuration file.')
    
    # Check num_classes from the correct location in config
    num_classes = config['model'].get('num_classes', config['val_dataset'].get('num_classes', 2))
    
    # Build transforms consistent with training
    val_transforms = []
    for t_cfg in val_dataset_config.get('transforms', []):
        t_type = t_cfg.get('type')
        if t_type == 'Resize':
            val_transforms.append(Resize(t_cfg.get('size', [256, 256])))
        elif t_type == 'Normalize':
            val_transforms.append(Normalize())
        elif t_type == 'SixValueLabelConvert' or t_type == 'FourValueLabelConvert':
            val_transforms.append(FourValueLabelConvert(
                extremely_high_weight=t_cfg.get('extremely_high_weight', 1.5),
                high_weight=t_cfg.get('high_weight', 0.5),
                background_weight=t_cfg.get('background_weight', 1.0),
                ignore_index=t_cfg.get('ignore_index', 255)
            ))
        elif t_type == 'BinaryLabelConvert':
            val_transforms.append(BinaryLabelConvert())
    
    val_transform = Compose(val_transforms)
    
    val_dataset = CrackDataset(
        data_root=val_dataset_config['dataset_root'],
        file_list=val_dataset_config.get('val_path', val_dataset_config.get('train_path')),
        transforms=[val_transform],
        mode='val',
        num_classes=num_classes
    )

    if len(val_dataset) == 0:
        raise ValueError('The length of val_dataset is 0. Please check if your dataset is valid')

    # Create model
    model_config = config['model']
    model = get_model(model_config['type'], **model_config.get('kwargs', {}))
    
    # Load model weights
    if args.model_path and os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Handle state dict loading for different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
            
        logger.info(f'Loaded trained params of model from {args.model_path}')
    else:
        logger.warning('Model path not provided or does not exist. Using randomly initialized weights.')

    model = model.to(device)

    # Perform evaluation
    miou, acc, class_iou, class_precision, class_recall, kappa = evaluate(
        model=model,
        eval_dataset=val_dataset,
        device=device,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        print_detail=True,
        batch_size=args.batch_size
    )

    return miou, acc, class_iou, class_precision, class_recall, kappa


if __name__ == '__main__':
    args = parse_args()
    main(args)

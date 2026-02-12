#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
from pathlib import Path

BACKGROUND = 0
EXTREMELY_HIGH = 255
HIGH = 204
VERY_LOW = 51

class PseudoLabelGenerator:
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.update_hard_thresh = getattr(args, 'update_hard_thresh', 0.9)
        self.update_start_ratio = getattr(args, 'update_start_ratio', 0.1)
        self.update_ratio_step = getattr(args, 'update_ratio_step', 0.15)
        self.update_max_ratio = getattr(args, 'update_max_ratio', 0.85)
        self.round_num = 0
    
    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
    
    def _read_train_list(self, list_path, dataset_root):
        samples = []
        if not os.path.exists(list_path):
            return samples
        with open(list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                img_path = parts[0]
                label_path = parts[1] if len(parts) > 1 else img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.png'
                samples.append({'img': img_path, 'label': label_path})
        return samples
    
    def generate_pseudo_labels(self, model, dataloader, device, dataset_root, train_list_path):
        self._log(f"\n开始生成伪标签（Round {self.round_num})...")
        dataset_root = Path(dataset_root)
        current_update_ratio = min(self.update_start_ratio + self.round_num * self.update_ratio_step, self.update_max_ratio)
        model.eval()
        original_samples = self._read_train_list(train_list_path, dataset_root)
        if len(original_samples) == 0: return None
        
        original_labels = {}
        uncertain_pixels_before = 0
        total_pixels = 0
        for sample in original_samples:
            label_path = dataset_root / sample['label']
            if not label_path.exists(): continue
            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if label is None: continue
            uncertain_mask = (label == VERY_LOW) | (label == HIGH) | (label == 128)
            uncertain_pixels_before += int(uncertain_mask.sum())
            total_pixels += label.size
            original_labels[str(label_path.resolve())] = label
        
        if total_pixels == 0: return None
        uncertain_ratio_before = uncertain_pixels_before / total_pixels
        self._log(f"总不确定像素: {uncertain_pixels_before / 1e6:.2f}M / {total_pixels / 1e6:.2f}M ({uncertain_ratio_before:.2%})")
        if uncertain_pixels_before == 0: return None
        
        predictions = {}
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                label_paths = batch.get('label_path', None)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                fg_prob = probs[:, 1, :, :]
                for i in range(len(images)):
                    if label_paths is None: continue
                    label_path = str(label_paths[i].item() if isinstance(label_paths[i], torch.Tensor) else label_paths[i])
                    label_path_norm = str(Path(label_path).resolve())
                    predictions[label_path_norm] = fg_prob[i].cpu().numpy()
        
        self._log(f"推理完成，收集了 {len(predictions)} 个样本的预测")
        
        label_dir = dataset_root / f'labels{self.round_num}'
        label_dir.mkdir(parents=True, exist_ok=True)
        updated_labels_dict = {}
        update_count = 0
        
        for label_path_norm, original_label in original_labels.items():
            updated_label = original_label.copy()
            if label_path_norm not in predictions:
                updated_labels_dict[label_path_norm] = updated_label
                continue
            fg_prob = predictions[label_path_norm]
            if fg_prob.shape != updated_label.shape:
                fg_prob = cv2.resize(fg_prob, (updated_label.shape[1], updated_label.shape[0]))
            uncertain_mask = (updated_label == VERY_LOW) | (updated_label == HIGH) | (updated_label == 128)
            uncertain_coords = np.where(uncertain_mask)
            if len(uncertain_coords[0]) == 0:
                updated_labels_dict[label_path_norm] = updated_label
                continue
            pixel_confs = [(fg_prob[uncertain_coords[0][idx], uncertain_coords[1][idx]], uncertain_coords[0][idx], uncertain_coords[1][idx]) for idx in range(len(uncertain_coords[0]))]
            pixel_confs.sort(key=lambda x: -x[0])
            sample_update_count = max(1, int(len(pixel_confs) * current_update_ratio))
            for idx in range(sample_update_count):
                conf, y, x = pixel_confs[idx]
                if conf >= self.update_hard_thresh:
                    updated_label[y, x] = EXTREMELY_HIGH
                elif conf >= 0.5:
                    updated_label[y, x] = HIGH
                update_count += 1
            updated_labels_dict[label_path_norm] = updated_label
        
        for label_path_norm, updated_label in updated_labels_dict.items():
            filename = os.path.basename(label_path_norm)
            output_path = label_dir / filename
            cv2.imwrite(str(output_path), updated_label)
        
        updated_samples = []
        for sample in original_samples:
            img_path = sample['img']
            old_label_path = sample['label']
            new_label_path = old_label_path.replace('labels', f'labels{self.round_num}') if 'labels' in old_label_path else f'labels{self.round_num}/' + os.path.basename(old_label_path)
            updated_samples.append({'img': img_path, 'label': new_label_path})
        
        list_dir = Path(train_list_path).parent
        new_list_path = list_dir / f'train_round{self.round_num}.txt'
        with open(new_list_path, 'w', encoding='utf-8') as f:
            for sample in updated_samples:
                f.write(f"{sample['img']} {sample['label']}\n")
        
        uncertain_pixels_after = 0
        for updated_label in updated_labels_dict.values():
            uncertain_mask = (updated_label == VERY_LOW) | (updated_label == HIGH) | (updated_label == 128)
            uncertain_pixels_after += int(uncertain_mask.sum())
        
        uncertain_ratio_after = uncertain_pixels_after / total_pixels
        self._log(f"总不确定像素: {uncertain_pixels_after / 1e6:.2f}M ({uncertain_ratio_after:.2%})")
        self._log(f"更新的像素数: {update_count}")
        self._log(f" 更新后的训练列表已保存: {new_list_path}")
        return str(new_list_path)

def update_pseudo_labels(model, train_loader, device, args, logger=None, round_num=None):
    if hasattr(args, 'opts_dict'):
        train_list_path = args.opts_dict.get('train_dataset.train_path')
        dataset_root = Path(args.opts_dict.get('train_dataset.dataset_root', '.'))
    else:
        return None
    generator = PseudoLabelGenerator(args, logger)
    if round_num is not None:
        generator.round_num = round_num
    return generator.generate_pseudo_labels(model, train_loader, device, dataset_root, train_list_path)

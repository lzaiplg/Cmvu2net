import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random
import logging

logger = logging.getLogger(__name__)


class CrackDataset(Dataset):
    """Custom dataset for crack segmentation."""
    
    def __init__(self, data_root, file_list, mode='train', num_classes=2, transforms=None):
        self.data_root = data_root
        self.mode = mode
        self.num_classes = num_classes
        self.transforms = transforms or []
        self.file_list = file_list
        
        raw_lines = []
        with open(file_list, 'r') as f:
            raw_lines = [line.strip() for line in f.readlines()]

        validated = []
        skipped = 0
        for line in raw_lines:
            if not line:
                continue
            parts = line.split()
            if len(parts) == 1:
                img_rel = parts[0]
                lbl_rel = img_rel.replace('images', 'labels').rsplit('.', 1)[0] + '.png'
            else:
                img_rel, lbl_rel = parts[0], parts[1]

            img_abs = os.path.join(self.data_root, img_rel)
            lbl_abs = os.path.join(self.data_root, lbl_rel)

            if os.path.exists(img_abs) and os.path.exists(lbl_abs):
                validated.append([img_rel, lbl_rel])
            else:
                skipped += 1
                logger.warning(f'[DATASET] 跳过缺失文件条目: img={img_abs} exists={os.path.exists(img_abs)}, label={lbl_abs} exists={os.path.exists(lbl_abs)}')
        
        if skipped > 0:
            logger.info(f'[DATASET] 从 {file_list} 中跳过 {skipped} 个缺失的条目，剩余 {len(validated)} 个样本')

        self.img_files = validated
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        line = self.img_files[idx]
        img_path = os.path.join(self.data_root, line[0])
        mask_path = os.path.join(self.data_root, line[1])
        
        image = cv2.imread(img_path)
        if image is None:
            logger.error(f"Failed to read image: {img_path}")
            raise ValueError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            logger.error(f"Failed to read label: {mask_path}")
            raise ValueError(f"Failed to read label: {mask_path}")
        
        semantic_weights = None
        for transform in self.transforms:
            image, label, semantic_weights = transform(image, label, semantic_weights)
        
        image = image.astype(np.float32) / 255.0
        label = label.astype(np.int64)
        
        image = np.transpose(image, (2, 0, 1))

        return {
            'image': image,
            'label': label,
            'semantic_weights': semantic_weights,
            'image_path': img_path,
            'label_path': mask_path
        }


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, label, semantic_weights=None):
        for t in self.transforms:
            image, label, semantic_weights = t(image, label, semantic_weights)
        return image, label, semantic_weights


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, label, semantic_weights=None):
        if random.random() < self.prob:
            image = np.fliplr(image)
            label = np.fliplr(label)
            if semantic_weights is not None:
                semantic_weights = np.fliplr(semantic_weights)
        return image, label, semantic_weights


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, label, semantic_weights=None):
        if random.random() < self.prob:
            image = np.flipud(image)
            label = np.flipud(label)
            if semantic_weights is not None:
                semantic_weights = np.flipud(semantic_weights)
        return image, label, semantic_weights


class RandomCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size
    
    def __call__(self, image, label, semantic_weights=None):
        h, w = image.shape[:2]
        ch, cw = self.crop_size
        
        if h > ch and w > cw:
            i = random.randint(0, h - ch)
            j = random.randint(0, w - cw)
            image = image[i:i+ch, j:j+cw]
            label = label[i:i+ch, j:j+cw]
            if semantic_weights is not None:
                semantic_weights = semantic_weights[i:i+ch, j:j+cw]
        else:
            if h < ch or w < cw:
                ph = max(0, ch - h)
                pw = max(0, cw - w)
                image = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode='constant')
                label = np.pad(label, ((0, ph), (0, pw)), mode='constant', constant_values=255)
                if semantic_weights is not None:
                    if semantic_weights.ndim == 3:
                        semantic_weights = np.pad(semantic_weights, ((0, ph), (0, pw), (0, 0)), mode='constant', constant_values=0.0)
                    else:
                        semantic_weights = np.pad(semantic_weights, ((0, ph), (0, pw)), mode='constant', constant_values=0.0)

        return image, label, semantic_weights


class Resize:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image, label, semantic_weights=None):
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.size, interpolation=cv2.INTER_NEAREST)
        if semantic_weights is not None:
            semantic_weights = cv2.resize(semantic_weights.astype('float32'), self.size, interpolation=cv2.INTER_NEAREST)
        return image, label, semantic_weights


class ResizeStepScaling:
    """类似PaddleSeg的ResizeStepScaling，实现动态尺寸缩放"""
    def __init__(self, min_scale_factor=0.5, max_scale_factor=2.0, scale_step_size=0.25):
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_step_size = scale_step_size
        
        self.scale_factors = []
        factor = self.min_scale_factor
        while factor <= self.max_scale_factor + 1e-5:
            self.scale_factors.append(factor)
            factor += self.scale_step_size

    def __call__(self, image, label, semantic_weights=None):
        scale_factor = random.choice(self.scale_factors)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if semantic_weights is not None:
            semantic_weights = cv2.resize(semantic_weights.astype('float32'), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return image, label, semantic_weights


class RandomPaddingCrop:
    """类似PaddleSeg的RandomPaddingCrop，实现随机填充裁剪"""
    def __init__(self, crop_size, cat_max_ratio=0.75, ignore_index=255):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def __call__(self, image, label, semantic_weights=None):
        h, w = image.shape[:2]
        crop_h, crop_w = self.crop_size
        
        if min(h, w) < crop_h:
            pad_h = max(crop_h - h, 0)
            pad_w = max(crop_w - w, 0)
            image = np.pad(image, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant', constant_values=0)
            label = np.pad(label, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)), mode='constant', constant_values=self.ignore_index)
            if semantic_weights is not None:
                if semantic_weights.ndim == 3:
                    semantic_weights = np.pad(semantic_weights, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant', constant_values=0.0)
                else:
                    semantic_weights = np.pad(semantic_weights, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)), mode='constant', constant_values=0.0)
            h, w = image.shape[:2]

        for _ in range(4): 
            y = np.random.randint(0, h - crop_h + 1)
            x = np.random.randint(0, w - crop_w + 1)

            image_crop = image[y:y+crop_h, x:x+crop_w, :]
            label_crop = label[y:y+crop_h, x:x+crop_w]
            semantic_weights_crop = None
            if semantic_weights is not None:
                semantic_weights_crop = semantic_weights[y:y+crop_h, x:x+crop_w]


            if self.cat_max_ratio < 1.0:
                labels, counts = np.unique(label_crop, return_counts=True)

                total_valid_pixels = sum(count for l, c in zip(labels, counts) if l != self.ignore_index)
                if total_valid_pixels > 0:
                    fg_ratio = sum(count for l, c in zip(labels, counts) if l != 0 and l != self.ignore_index) / total_valid_pixels
                    if fg_ratio > self.cat_max_ratio:
                        continue
            break

        return image_crop, label_crop, semantic_weights_crop if semantic_weights is not None else (image_crop, label_crop, None)


class RandomDistort:
    def __init__(self, brightness_range=0.5, contrast_range=0.5, saturation_range=0.5, hue_range=0.5):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

    def __call__(self, image, label, semantic_weights=None):
        if random.random() < 0.5:
            brightness_factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
            image = np.clip(image * brightness_factor, 0.0, 1.0).astype(np.float32)


        if random.random() < 0.5:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray_mean = np.mean(gray) / 255.0
            contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
            image = np.clip((image - gray_mean) * contrast_factor + gray_mean, 0.0, 1.0).astype(np.float32)

 
        if random.random() < 0.5:
            hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            saturation_factor = 1.0 + random.uniform(-self.saturation_range, self.saturation_range)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    
        if random.random() < 0.5:
            hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hue_factor = random.uniform(-self.hue_range, self.hue_range)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_factor * 180) % 180
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        return image, label, semantic_weights


class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(self, image, label, semantic_weights=None):
        image = (image - self.mean) / self.std
        return image.astype(np.float32), label, semantic_weights


class BinaryLabelConvert:
    """Convert multi-level labels (0/255 -> foreground/background)."""
    def __call__(self, image, label, semantic_weights=None):
        binary_label = np.where(label == 255, 255, np.where(label == 0, 0, 1))
        return image, binary_label, semantic_weights


class FourValueLabelConvert:

    def __init__(self, extremely_high_weight=1.5, high_weight=0.5, background_weight=1.0, ignore_index=255,
                 ce_weights=None, dice_weights=None):
        self.extremely_high_weight = extremely_high_weight
        self.high_weight = high_weight
        self.background_weight = background_weight
        self.ignore_index = ignore_index
        
        self.ce_weights = ce_weights
        self.dice_weights = dice_weights
        
        if self.ce_weights is not None and self.dice_weights is not None:
            self.use_multi_channel = True
        else:
            self.use_multi_channel = False

        if self.ce_weights is not None and len(self.ce_weights) >= 3:
            self.extremely_high_weight = self.ce_weights[0]
            self.high_weight = self.ce_weights[1]
            self.background_weight = self.ce_weights[2]

    def _get_weight_map(self, label, weights):
        """Helper to generate weight map from label and weights dict."""
        ext_w, high_w, bg_w = weights
        sw = np.zeros_like(label, dtype=np.float32)
        
        has_multilevel = (51 in np.unique(label)) or (204 in np.unique(label))

        if has_multilevel:
            sw[label == 51] = 0.0
            sw[label == 204] = float(high_w)
            sw[label == 255] = float(ext_w)
            sw[label == 0] = float(bg_w)
        else:
            sw[label == 0] = float(bg_w)
            sw[label == 255] = float(ext_w)
            
        return sw

    def __call__(self, image, label, semantic_weights=None):
        if self.use_multi_channel:
            sw_ce = self._get_weight_map(label, self.ce_weights)
            sw_dice = self._get_weight_map(label, self.dice_weights)
            sw = np.stack([sw_ce, sw_dice], axis=-1)
        else:
            sw = self._get_weight_map(label, [self.extremely_high_weight, self.high_weight, self.background_weight])

        new_label = np.copy(label)
        has_multilevel = (51 in np.unique(label)) or (204 in np.unique(label))
        if has_multilevel:
            new_label[label == 51] = self.ignore_index
            new_label[label == 204] = 1
            new_label[label == 255] = 1
            new_label[label == 0] = 0
        else:
            new_label[label == 255] = 1
            new_label[label == 0] = 0
            
        mask_valid = (new_label == 0) | (new_label == 1) | (new_label == self.ignore_index)
        if not np.all(mask_valid):
            new_label[~mask_valid] = self.ignore_index
            if self.use_multi_channel:
                sw[~mask_valid, :] = 0.0
            else:
                sw[~mask_valid] = 0.0

        return image, new_label, sw


class CustomCollate:
    """Custom collate function class to handle variable-sized inputs (picklable for multiprocessing)."""
    
    def __call__(self, batch):
        import torch
        images = torch.stack([torch.from_numpy(item['image']).float() for item in batch])
        labels = torch.stack([torch.from_numpy(item['label']).long() for item in batch])
        if 'semantic_weights' in batch[0]:
            sems = [item.get('semantic_weights', None) for item in batch]
            sems = [s if s is not None else np.ones_like(batch[0]['label'], dtype=np.float32) for s in sems]
            semantic_weights = torch.stack([torch.from_numpy(s).float() for s in sems])
        else:
            semantic_weights = torch.ones_like(labels).float()
        label_paths = [item['label_path'] for item in batch]
        image_paths = [item['image_path'] for item in batch]
        return {
            'image': images,
            'label': labels,
            'semantic_weights': semantic_weights,
            'label_path': label_paths,
            'image_path': image_paths,
        }


def get_collate_fn():
    """Custom collate function to handle variable-sized inputs."""
    return CustomCollate()


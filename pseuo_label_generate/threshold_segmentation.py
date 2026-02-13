import os
import cv2
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image

def safe_imread(file_path, grayscale=True):
    """安全读取图像文件，避免libpng警告"""
    try:
        if grayscale:
            with Image.open(file_path) as img:
                if img.mode != 'L':
                    img = img.convert('L')
                return np.array(img)
        else:
            with Image.open(file_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return np.array(img)
    except Exception as e:
        print(f"警告: 无法读取文件 {file_path}: {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(
        description="批量自适应阈值裂缝分割，生成二值化掩码或四值置信度掩码")
    parser.add_argument("--img_dir", type=str,
                        default=os.path.join("..", "data", "ConcreteData", "Train_image"),
                        help="待分割图像文件夹")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join("..", "data", "ConcreteData", "threshold_label"),
                        help="掩码保存文件夹")
    parser.add_argument("--trimap_dir", type=str,
                        default=os.path.join("..", "data", "ConcreteData", "Layercam", "aug", "cam"),
                        help="CAM 三值掩码目录 (0/128/255)。若提供则在255区域内进行Sauvola多k分割输出四值掩码")
    parser.add_argument("--method", type=str, default="sauvola",
                        choices=["otsu", "sauvola", "niblack", "adaptive_mean"],
                        help="阈值算法")
    parser.add_argument("--window_size", type=int, default=79,
                        help="Sauvola/Niblack 窗口大小，应为奇数")
    parser.add_argument("--k", type=float, default=0.1,
                        help="Niblack/Sauvola 参数 k (用于二值掩码生成)")
    parser.add_argument("--k_high", type=float, default=0.3, 
                        help="高置信区域的k值 (用于四值掩码生成)")
    parser.add_argument("--k_mid", type=float, default=0.2, 
                        help="中置信区域的k值 (用于四值掩码生成)")
    parser.add_argument("--k_low", type=float, default=0.1, 
                        help="低置信区域的k值 (用于四值掩码生成)")
    parser.add_argument("--min_area", type=int, default=100,
                        help="去除小连通域面积阈值")
    parser.add_argument("--area_divisor", type=float, default=10.0,
                        help="面积阈值分母，阈值=max(最大连通域面积/area_divisor, min_area_floor)")
    parser.add_argument("--min_area_floor", type=float, default=50.0,
                        help="面积阈值下限，参与阈值=max(最大连通域面积/area_divisor, min_area_floor)")
    parser.add_argument("--ar_threshold", type=float, default=4.0,
                        help="骨架长宽比阈值，用于双重长宽比过滤逻辑")
    parser.add_argument("--bbox_ar_threshold", type=float, default=2.0,
                        help="外接矩形长宽比阈值，外接矩形长宽比>=此值才考虑骨架长宽比")
    parser.add_argument("--gt_dir", type=str,
                        default=os.path.join("..", "data", "ConcreteData", "Train_label"),
                        help="真实掩码目录 (可选)，若提供则计算指标")
    parser.add_argument("--metrics_csv", type=str, default="metrics_threshold_segmentation.csv",
                        help="指标 CSV 输出文件名 (保存到 save_dir 下)")
    parser.add_argument("--cam_dir", type=str,
                        default=os.path.join("..", "data", "ConcreteData", "Layercam", "aug", "cam_post"),
                        help="CAM 掩码目录 (与阈值分割结果取交集，若为空则跳过)")
    return parser.parse_args()

def _threshold(img_gray: np.ndarray, method: str, window_size: int, k: float):
    """返回 0/255 二值掩码"""
    if method == "otsu":
        _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    if method == "adaptive_mean":
        mask = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, max(3, window_size)|1, 0)
        return mask

    if not _HAS_SKIMAGE:
        raise RuntimeError("当前环境缺少 scikit-image，无法使用 {} 方法".format(method))

    if method == "sauvola":
        thresh = threshold_sauvola(img_gray, window_size=window_size, k=k)
    elif method == "niblack":
        thresh = threshold_niblack(img_gray, window_size=window_size, k=k)
    else:
        raise ValueError("未知方法: " + method)

    mask = (img_gray > thresh).astype(np.uint8) * 255
    mask = 255 - mask
    return mask

def _postprocess(mask: np.ndarray, min_area: int):
    """形态学开闭 + 连通域过滤"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            keep[labels == i] = 255
    return keep

def _adaptive_connected_components_filter(mask: np.ndarray, area_divisor: float = 10.0, min_area_floor: float = 100.0, ar_threshold: float = 4.0, bbox_ar_threshold: float = 2.0):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:
        return mask
    
    areas = []
    bbox_aspect_ratios = []
    skeleton_aspect_ratios = []
    valid_indices = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        left = int(stats[i, cv2.CC_STAT_LEFT])
        top = int(stats[i, cv2.CC_STAT_TOP])
        width = int(stats[i, cv2.CC_STAT_WIDTH])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])

        bbox_aspect_ratio = max(width, height) / (min(width, height) + 1e-6)

        comp_crop = (labels[top:top+height, left:left+width] == i)

        if skeletonize is not None:
            try:
                skel = skeletonize(comp_crop.astype(np.uint8) > 0)
                skeleton_length = int(np.count_nonzero(skel))
            except Exception:
                skeleton_length = int(max(width, height))
        else:
            skeleton_length = int(max(width, height))

        if skeleton_length <= 0:
            skeleton_length = 1

        skeleton_aspect_ratio = (skeleton_length * skeleton_length) / (area + 1e-6)

        areas.append(area)
        bbox_aspect_ratios.append(bbox_aspect_ratio)
        skeleton_aspect_ratios.append(skeleton_aspect_ratio)
        valid_indices.append(i)

    if not areas:
        return mask

    max_area = float(max(areas))
    threshold = max(max_area / float(area_divisor), float(min_area_floor))

    print(f"    双重长宽比连通域过滤: 最大面积={int(max_area)}, 面积阈值={threshold:.3f} (max(max_area/{area_divisor}, {min_area_floor}))")
    print(f"    外接矩形长宽比阈值={bbox_ar_threshold}, 骨架长宽比阈值={ar_threshold}")

    keep = np.zeros_like(mask)
    sorted_data = sorted(zip(areas, bbox_aspect_ratios, skeleton_aspect_ratios, valid_indices), key=lambda x: x[0], reverse=True)
    
    for idx, (area, bbox_ar, skel_ar, label_idx) in enumerate(sorted_data, start=1):
        if area >= threshold:
            keep[labels == label_idx] = 255
            print(f"      保留连通域{idx}: 面积={area} >= {threshold:.3f} (面积足够大)")
        else:
            if bbox_ar < float(bbox_ar_threshold):
                print(f"      过滤连通域{idx}: 面积={area} < {threshold:.3f} 且 外接矩形长宽比={bbox_ar:.3f} < {bbox_ar_threshold}")
            elif skel_ar < float(ar_threshold):
                print(f"      过滤连通域{idx}: 面积={area} < {threshold:.3f}, 外接矩形长宽比={bbox_ar:.3f} >= {bbox_ar_threshold} 但 骨架长宽比={skel_ar:.3f} < {ar_threshold}")
            else:
                keep[labels == label_idx] = 255
                print(f"      保留连通域{idx}: 面积={area} < {threshold:.3f} 但 外接矩形长宽比={bbox_ar:.3f} >= {bbox_ar_threshold} 且 骨架长宽比={skel_ar:.3f} >= {ar_threshold}")

    return keep

def calculate_metrics_np(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() > 0 and gt.sum() > 0:
        tp = np.sum(pred * gt)
        precision = tp / pred.sum()
        recall = tp / gt.sum()
        dice = 2 * tp / (pred.sum() + gt.sum())
        union = pred.sum() + gt.sum() - tp
        iou = tp / union
        return dice, precision, recall, iou
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 1.0, 1.0, 1.0
    else:
        return 0.0, 0.0, 0.0, 0.0

def find_gt_mask(name_without_ext: str, gt_dir: Path):
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
        candidate = gt_dir / f"{name_without_ext}{ext}"
        if candidate.exists():
            return safe_imread(str(candidate), grayscale=True)
    return None

def find_cam_mask(name_without_ext: str, cam_dir: Path):
    """在给定目录中按多种后缀寻找同名 CAM 掩码"""
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
        candidate = cam_dir / f"{name_without_ext}{ext}"
        if candidate.exists():
            return safe_imread(str(candidate), grayscale=True)
    return None

def main():
    args = parse_args()
    img_dir = Path(args.img_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")]

    all_metrics = []

    for img_path in tqdm(img_files, desc="processing"):
        img = safe_imread(str(img_path), grayscale=True)
        if img is None:
            print(f"读取失败: {img_path}")
            continue

        if args.trimap_dir:
            trimap = find_cam_mask(img_path.stem, Path(args.trimap_dir))
            if trimap is not None:
                high_region = (trimap == 255)

                if args.method != "sauvola":
                    method_for_quad = "sauvola"
                else:
                    method_for_quad = args.method

                if not _HAS_SKIMAGE and method_for_quad in ("sauvola", "niblack"):
                    raise RuntimeError("缺少 scikit-image，无法进行 Sauvola/Niblack 分割以生成四值掩码")

                if method_for_quad == "sauvola":
                    t_high = threshold_sauvola(img, window_size=args.window_size, k=args.k_high)
                    t_mid = threshold_sauvola(img, window_size=args.window_size, k=args.k_mid)
                    t_low = threshold_sauvola(img, window_size=args.window_size, k=args.k_low)
                else:
                    t_high = threshold_niblack(img, window_size=args.window_size, k=args.k_high)
                    t_mid = threshold_niblack(img, window_size=args.window_size, k=args.k_mid)
                    t_low = threshold_niblack(img, window_size=args.window_size, k=args.k_low)

                m_high = (255 - ((img > t_high).astype(np.uint8) * 255)) > 0
                m_mid = (255 - ((img > t_mid).astype(np.uint8) * 255)) > 0
                m_low = (255 - ((img > t_low).astype(np.uint8) * 255)) > 0

                m_high = m_high & high_region
                m_mid = m_mid & high_region
                m_low = m_low & high_region

                print(f"    步骤1: k={args.k_low}自适应连通域过滤")
                m_low_cleaned = _adaptive_connected_components_filter(
                    m_low.astype(np.uint8) * 255,
                    area_divisor=args.area_divisor,
                    min_area_floor=args.min_area_floor,
                    ar_threshold=args.ar_threshold,
                    bbox_ar_threshold=args.bbox_ar_threshold
                ) > 0
                
                print(f"    步骤2: k={args.k_mid}过滤，仅保留与k={args.k_low}重合部分")
                m_mid_filtered = m_mid & m_low_cleaned
                
                print(f"    步骤3: k={args.k_high}过滤，仅保留与k={args.k_mid}重合部分")
                m_high_filtered = m_high & m_mid_filtered
                
                high_conf = m_high_filtered
                mid_conf = m_mid_filtered & (~m_high_filtered)
                low_conf = m_low_cleaned & (~m_mid_filtered)

                quad_mask = np.zeros_like(img, dtype=np.uint8)
                quad_mask[low_conf] = 64
                quad_mask[mid_conf] = 128
                quad_mask[high_conf] = 255

                low_count = np.sum(low_conf)
                mid_count = np.sum(mid_conf)
                high_count = np.sum(high_conf)
                total_region = np.sum(high_region)
                
                low_cleaned_count = np.sum(m_low_cleaned)
                mid_filtered_count = np.sum(m_mid_filtered)
                high_filtered_count = np.sum(m_high_filtered)
                
                print(f"{img_path.name}: 层次化构建 - 低置信={low_count}, 中置信={mid_count}, 高置信={high_count}")
                print(f"  中间结果: k={args.k_low}清理后={low_cleaned_count}, k={args.k_mid}过滤后={mid_filtered_count}, k={args.k_high}过滤后={high_filtered_count}")
                print(f"  CAM高响应区域={total_region}")

                save_path = save_dir / (img_path.stem + ".png")
                cv2.imwrite(str(save_path), quad_mask)

                if args.gt_dir:
                    gt_mask = find_gt_mask(img_path.stem, Path(args.gt_dir))
                    if gt_mask is not None and gt_mask.shape == quad_mask.shape:
                        _, gt_bin = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)
                        
                        high_conf_mask = (quad_mask == 255).astype(np.uint8)
                        dice_high, precision_high, recall_high, iou_high = calculate_metrics_np(high_conf_mask, gt_bin)
                        
                        high_mid_mask = ((quad_mask == 255) | (quad_mask == 128)).astype(np.uint8)
                        dice_high_mid, precision_high_mid, recall_high_mid, iou_high_mid = calculate_metrics_np(high_mid_mask, gt_bin)
                        
                        all_conf_mask = ((quad_mask == 255) | (quad_mask == 128) | (quad_mask == 64)).astype(np.uint8)
                        dice_all, precision_all, recall_all, iou_all = calculate_metrics_np(all_conf_mask, gt_bin)
                        
                        all_metrics.append({
                            "filename": img_path.name,
                            "confidence_level": "high_only",
                            "dice": dice_high,
                            "precision": precision_high,
                            "recall": recall_high,
                            "iou": iou_high
                        })
                        
                        all_metrics.append({
                            "filename": img_path.name,
                            "confidence_level": "high_mid",
                            "dice": dice_high_mid,
                            "precision": precision_high_mid,
                            "recall": recall_high_mid,
                            "iou": iou_high_mid
                        })
                        
                        all_metrics.append({
                            "filename": img_path.name,
                            "confidence_level": "all_levels",
                            "dice": dice_all,
                            "precision": precision_all,
                            "recall": recall_all,
                            "iou": iou_all
                        })
                        
                        print(f"  评估指标:")
                        print(f"    仅高置信: Dice={dice_high:.3f}, IoU={iou_high:.3f}, Precision={precision_high:.3f}, Recall={recall_high:.3f}")
                        print(f"    高+中置信: Dice={dice_high_mid:.3f}, IoU={iou_high_mid:.3f}, Precision={precision_high_mid:.3f}, Recall={recall_high_mid:.3f}")
                        print(f"    全部置信: Dice={dice_all:.3f}, IoU={iou_all:.3f}, Precision={precision_all:.3f}, Recall={recall_all:.3f}")
                        
                    elif gt_mask is None:
                        print(f"  未找到对应的GT掩码")
                    elif gt_mask.shape != quad_mask.shape:
                        print(f"  GT掩码尺寸不匹配: GT={gt_mask.shape}, Pred={quad_mask.shape}")
                
                continue
            else:
                print(f"跳过图像 {img_path.name}：未找到对应的CAM三值掩码")
                continue

        mask_thresh = _threshold(img, args.method, args.window_size, args.k)

        mask_for_post = mask_thresh
        skip_this_image = False
        if args.cam_dir:
            cam_mask_raw = find_cam_mask(img_path.stem, Path(args.cam_dir))
            if cam_mask_raw is not None:
                if np.sum(cam_mask_raw) == 0:
                    skip_this_image = True
                else:
                    if cam_mask_raw.shape != mask_thresh.shape:
                        cam_mask_raw = cv2.resize(cam_mask_raw, (mask_thresh.shape[1], mask_thresh.shape[0]), interpolation=cv2.INTER_NEAREST)
                    cam_bin = (cam_mask_raw > 0).astype(np.uint8)
                    thresh_bin = (mask_thresh > 0).astype(np.uint8)
                    mask_for_post = (cam_bin & thresh_bin).astype(np.uint8) * 255
        
        if skip_this_image:
            continue

        mask = _postprocess(mask_for_post, args.min_area)

        save_path = save_dir / (img_path.stem + ".png")
        cv2.imwrite(str(save_path), mask)

        if args.gt_dir:
            gt_mask = find_gt_mask(img_path.stem, Path(args.gt_dir))
            if gt_mask is not None and gt_mask.shape == mask.shape:
                _, gt_bin = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)
                dice, precision, recall, iou = calculate_metrics_np(mask, gt_bin)
                all_metrics.append({
                    "filename": img_path.name,
                    "dice": dice,
                    "precision": precision,
                    "recall": recall,
                    "iou": iou
                })

    print(f"处理完毕，掩码已保存到: {save_dir}")

    if all_metrics:
        import pandas as pd

        metrics_df = pd.DataFrame(all_metrics)
        
        if 'confidence_level' in metrics_df.columns:
            print("\n------ 四值掩码层次化评估摘要 ------")
            
            for level in ['high_only', 'high_mid', 'all_levels']:
                level_data = metrics_df[metrics_df['confidence_level'] == level]
                if len(level_data) > 0:
                    mean_dice = level_data['dice'].mean()
                    std_dice = level_data['dice'].std()
                    mean_precision = level_data['precision'].mean()
                    std_precision = level_data['precision'].std()
                    mean_recall = level_data['recall'].mean()
                    std_recall = level_data['recall'].std()
                    mean_iou = level_data['iou'].mean()
                    std_iou = level_data['iou'].std()
                    
                    level_name = {
                        'high_only': '仅高置信区域',
                        'high_mid': '高+中置信区域', 
                        'all_levels': '全部置信区域'
                    }[level]
                    
                    print(f"{level_name} ({len(level_data)}个样本):")
                    print(f"  平均 Dice     : {mean_dice:.4f} ± {std_dice:.4f}")
                    print(f"  平均 Precision: {mean_precision:.4f} ± {std_precision:.4f}")
                    print(f"  平均 Recall   : {mean_recall:.4f} ± {std_recall:.4f}")
                    print(f"  平均 IoU      : {mean_iou:.4f} ± {std_iou:.4f}")
                    print()
        else:
            mean_dice = metrics_df['dice'].mean()
            std_dice = metrics_df['dice'].std()
            mean_precision = metrics_df['precision'].mean()
            std_precision = metrics_df['precision'].std()
            mean_recall = metrics_df['recall'].mean()
            std_recall = metrics_df['recall'].std()
            mean_iou = metrics_df['iou'].mean()
            std_iou = metrics_df['iou'].std()

            print("\n------ 二值掩码评估摘要 ------")
            print(f"处理文件数量: {len(metrics_df)}")
            print(f"平均 Dice     : {mean_dice:.4f} ± {std_dice:.4f}")
            print(f"平均 Precision: {mean_precision:.4f} ± {std_precision:.4f}")
            print(f"平均 Recall   : {mean_recall:.4f} ± {std_recall:.4f}")
            print(f"平均 IoU      : {mean_iou:.4f} ± {std_iou:.4f}")
            print("--------------------------\n")

        csv_path = save_dir / args.metrics_csv
        metrics_df.to_csv(csv_path, index=False)
        print(f"详细指标已保存至: {csv_path.resolve()}")
    else:
        print("未计算指标 (可能未提供 GT 目录、未匹配任何真实掩码，或所有图像都跳过了)。")

if __name__ == "__main__":
    main() 
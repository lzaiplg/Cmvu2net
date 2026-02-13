
import os
import pandas as pd
import torch
import cv2
from tqdm import tqdm
import glob
from PIL import Image
import numpy as np
from torchvision import models, transforms
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn
from skimage import measure
from scipy import ndimage
import argparse


def largest_connected_component(binary_img, ratio=1):
    labeled_img, num = ndimage.label(binary_img)
    
    if num <= 1:
        return binary_img
    
    sizes = ndimage.sum(binary_img, labeled_img, range(1, num+1))
    max_size = sizes.max()
    
    mask = np.zeros_like(binary_img, dtype=bool)
    for i, size in enumerate(sizes, 1):
        if size >= max_size / ratio:
            mask = mask | (labeled_img == i)
    
    return mask.astype(np.uint8)


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    """
    将 CAM 归一化到 [0, 1]
    """
    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max - cam_min < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return cam.astype(np.float32)


def cvs_smooth(cam: np.ndarray, kernel_size: int = 9) -> np.ndarray:

    cam = cam.astype(np.float32)
    k = (kernel_size, kernel_size)
    mean = cv2.blur(cam, k)
    sqmean = cv2.blur(cam * cam, k)
    var = np.maximum(sqmean - mean * mean, 0.0).astype(np.float32)
    std = np.sqrt(var + 1e-12)
    cv = std / (mean + 1e-6)

    smooth = cv2.bilateralFilter((cam * 255).astype(np.uint8), d=7, sigmaColor=20, sigmaSpace=7)
    smooth = smooth.astype(np.float32) / 255.0

    cv_norm = _normalize_cam(cv)
    cam_cvs = (1.0 - cv_norm) * cam + cv_norm * smooth
    return _normalize_cam(cam_cvs)


def generate_trimap_from_cam(cam: np.ndarray, fg_threshold: float = 0.15, bg_threshold: float = 0.1, lcc_ratio: int = 5, binary_threshold: float = 0.1) -> np.ndarray:
    cam = cam.astype(np.float32)
    
    if cam.size == 0 or np.all(cam <= 1e-6):
        return np.zeros_like(cam, dtype=np.uint8)
    
    base_mask = (cam > binary_threshold).astype(np.uint8)
    base_mask = largest_connected_component(base_mask, ratio=lcc_ratio)
    
    fg = (cam >= fg_threshold).astype(np.uint8) & base_mask
    
    tri = np.full(cam.shape, 0, dtype=np.uint8)
    tri[base_mask == 1] = 128
    tri[fg == 1] = 255
    
    return tri


def get_cam(filenames, model, output_dir, device, target_size=(448, 448), threshold=0.1,
            enable_cvs: bool = True, enable_ppg: bool = True, fg_threshold: float = 0.15, bg_threshold: float = 0.1, lcc_ratio: int = 5):
    """
    为混凝土裂缝图像生成LayerCAM
    
    Args:
        filenames: 图像文件路径列表
        model: 训练好的ResNet101模型
        output_dir: 输出目录
        device: 运行设备
        target_size: 输入到模型的图像大小
        threshold: CAM阈值
    """
    os.makedirs(os.path.join(output_dir, 'cam'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'vis_cam'), exist_ok=True)
    if enable_ppg:
        os.makedirs(os.path.join(output_dir, 'cam_trimap'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'vis_trimap'), exist_ok=True)
    
    target_layers = [model.layer4[-1]]
    
    cam = LayerCAM(model=model, target_layers=target_layers)
    
    for filename in tqdm(filenames, ncols=70):
        corrected_filename = filename.replace('\\', '/')
        image = cv2.imread(corrected_filename)
        if image is None:
            print(f"无法读取图像: {corrected_filename} (原始路径: {filename})")
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        
        is_crack = 1 if base_name.startswith("crack_") else 0
        
        image_resized = cv2.resize(image, target_size)
        
        input_tensor = preprocess(Image.fromarray(image_resized)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, dim=1)
            pred_class = preds.cpu().numpy()[0]
            
        image_norm = image_resized.astype(np.float32) / 255.0
        
        if pred_class == 1:
            targets = [ClassifierOutputTarget(1)]
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            if grayscale_cam.shape != (target_size[1], target_size[0]):
                grayscale_cam = cv2.resize(grayscale_cam, target_size)
            
            grayscale_cam = _normalize_cam(grayscale_cam)

            if enable_cvs:
                grayscale_cam = cvs_smooth(grayscale_cam)

            visualization = show_cam_on_image(image_norm, grayscale_cam, use_rgb=True)
            
            cam_binary = (grayscale_cam > float(threshold)).astype(np.uint8)
            
            cam_binary = largest_connected_component(cam_binary, ratio=lcc_ratio)
            
            cam_binary_visual = np.zeros_like(image_resized)
            cam_binary_visual[:, :, 0] = cam_binary * 255
            
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            axs[0].imshow(image_resized)
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            
            axs[1].imshow(visualization)
            axs[1].set_title('LayerCAM')
            axs[1].axis('off')
            
            axs[2].imshow(image_resized)
            axs[2].imshow(cam_binary_visual, alpha=0.5)
            axs[2].set_title('Binary Mask')
            axs[2].axis('off')
            
            plt.tight_layout()
            
            vis_save_path = os.path.join(output_dir, 'vis_cam', f'{name_without_ext}.png')
            plt.savefig(vis_save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            cam_save_path = os.path.join(output_dir, 'cam', f'{name_without_ext}.png')
            cv2.imwrite(cam_save_path, cam_binary * 255)

            if enable_ppg:
                tri_mask = generate_trimap_from_cam(grayscale_cam, fg_threshold=fg_threshold, bg_threshold=bg_threshold, lcc_ratio=lcc_ratio, binary_threshold=threshold)
                tri_save_path = os.path.join(output_dir, 'cam_trimap', f'{name_without_ext}.png')
                cv2.imwrite(tri_save_path, tri_mask)

                color = np.zeros_like(image_resized)
                color[tri_mask == 255] = (255, 0, 0)
                color[tri_mask == 128] = (0, 255, 0)
                fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
                axs2[0].imshow(image_resized)
                axs2[0].set_title('Original Image')
                axs2[0].axis('off')
                axs2[1].imshow(image_resized)
                axs2[1].imshow(color, alpha=0.4)
                axs2[1].set_title('PPG Trimap (FG=Red, IGN=Green)')
                axs2[1].axis('off')
                plt.tight_layout()
                tri_vis_path = os.path.join(output_dir, 'vis_trimap', f'{name_without_ext}.png')
                plt.savefig(tri_vis_path, bbox_inches='tight', pad_inches=0)
                plt.close()
            
        else:
            cam_binary = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
            cam_save_path = os.path.join(output_dir, 'cam', f'{name_without_ext}.png')
            cv2.imwrite(cam_save_path, cam_binary)


def find_best_binary_threshold(cam_dir, data_dir="../data/ConcreteData", sample_size=10):
    """
    寻找最佳的CAM二值化阈值，使用precision和recall的平均值作为评估指标
    
    Args:
        cam_dir: CAM图像目录
        data_dir: 数据根目录
        sample_size: 用于评估的样本数量
        
    Returns:
        最佳阈值
    """
    cam_files = sorted(glob.glob(os.path.join(cam_dir, "vis_cam", "*.png")))
    if not cam_files:
        print(f"在{cam_dir}/vis_cam/中未找到图像")
        return 0.5
    
    if len(cam_files) > sample_size:
        np.random.seed(42)
        cam_files = np.random.choice(cam_files, sample_size, replace=False).tolist()
    
    best_threshold = 0.0
    best_score = 0.0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        total_precision = 0.0
        total_recall = 0.0
        count = 0
        
        for cam_file in cam_files:
            base_name = os.path.basename(cam_file)
            image_id = os.path.splitext(base_name)[0]
            
            if not image_id.startswith("crack_"):
                continue
                
            vis_cam = plt.imread(cam_file)
            if vis_cam.shape[2] == 4:
                vis_cam = vis_cam[:, :, :3]
                
            cam_gray = np.mean(vis_cam[:, vis_cam.shape[1]//3:(vis_cam.shape[1]//3)*2, :], axis=2)
            
            mask_path = os.path.join(data_dir, "Train_label", f"{image_id}.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(data_dir, "Train_label", f"{image_id}.jpg")
                
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (cam_gray.shape[1], cam_gray.shape[0]))
                mask = (mask > 0).astype(np.uint8)
                
                cam_binary = (cam_gray > threshold).astype(np.uint8)
                
                tp = np.logical_and(cam_binary, mask).sum()
                fp = np.logical_and(cam_binary, 1 - mask).sum()
                fn = np.logical_and(1 - cam_binary, mask).sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                total_precision += precision
                total_recall += recall
                count += 1
        
        avg_precision = total_precision / count if count > 0 else 0
        avg_recall = total_recall / count if count > 0 else 0
        avg_score = (avg_precision + avg_recall) / 2
        
        print(f"阈值: {threshold:.2f}, 平均Precision: {avg_precision:.4f}, 平均Recall: {avg_recall:.4f}, 平均分数: {avg_score:.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_threshold = threshold
    
    print(f"最佳阈值: {best_threshold:.2f}, 最佳平均分数: {best_score:.4f}")
    return best_threshold


def process_cam_directory(cam_dir, output_dir, threshold=0.5):
    """
    处理CAM目录中的所有图像，应用二值化和后处理
    
    Args:
        cam_dir: CAM图像目录
        output_dir: 输出目录
        threshold: 二值化阈值
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cam_files = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
    
    for cam_file in tqdm(cam_files, desc="处理CAM", ncols=70):
        cam_image = cv2.imread(cam_file, cv2.IMREAD_GRAYSCALE)
        if cam_image is None:
            continue
            
        cam_binary = (cam_image > threshold * 255).astype(np.uint8)
        
        cam_binary = largest_connected_component(cam_binary, ratio=5)
        
        base_name = os.path.basename(cam_file)
        output_file = os.path.join(output_dir, base_name)
        cv2.imwrite(output_file, cam_binary * 255)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, choices=["raw", "aug", "refined"], default='aug', help='模型阶段')
    parser.add_argument('--data_dir', type=str, default='../data/ConcreteData', help='数据目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--threshold', type=float, default=None, help='CAM二值化阈值，如果不指定则自动寻找最佳阈值')
    parser.add_argument('--enable_cvs', action='store_true', help='启用CVS平滑')
    parser.add_argument('--enable_ppg', action='store_true', help='启用PPG三值掩码输出')
    parser.add_argument('--fg_threshold', type=float, default=0.3, help='三值掩码前景阈值（大于此值为前景）')
    parser.add_argument('--bg_threshold', type=float, default=0.1, help='三值掩码背景阈值（小于此值为背景）')
    parser.add_argument('--lcc_ratio', type=int, default=5, help='连通域保留比（与原先一致）')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    model_path = f'../resnet50_model/resnet50_{args.stage}_final.pth'
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        exit(1)
    
    if device.type == 'cuda':
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    model = model.to(device)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_dir = os.path.join(data_dir, "Train_image")
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录 {image_dir} 不存在")
        exit(1)
    filenames = sorted(glob.glob(os.path.join(image_dir, "*.jpg"))) + \
                sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not filenames:
        print(f"错误: 在 {image_dir} 下未找到图像文件")
        exit(1)
    output_dir = os.path.join(data_dir, "Layercam", args.stage)
    os.makedirs(output_dir, exist_ok=True)
    get_cam(
        filenames, model, output_dir, device,
        enable_cvs=args.enable_cvs,
        enable_ppg=args.enable_ppg,
        fg_threshold=args.fg_threshold,
        bg_threshold=args.bg_threshold,
        lcc_ratio=args.lcc_ratio
    )
    if args.threshold is None:
        threshold = find_best_binary_threshold(output_dir, data_dir)
    else:
        threshold = args.threshold
    cam_dir = os.path.join(output_dir, "cam")
    cam_post_dir = os.path.join(output_dir, "cam_post")
    process_cam_directory(cam_dir, cam_post_dir, threshold)
    print(f"LayerCAM生成完成，结果保存在 {output_dir}")
    print(f"使用的阈值: {threshold}")




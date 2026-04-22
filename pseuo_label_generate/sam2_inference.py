# mask不一样时,异常值检测

import sys
import os
# 解决 Windows 下多重 OpenMP 运行时冲突导致的 "libiomp5md.dll already initialized" 错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# 添加sam2到Python路径
sam_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sam2-main'))
sys.path.append(sam_path)

import time
import SimpleITK as sitk
import numpy as np
import torch
import cv2
import argparse
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
from PIL import Image

# 解决 NumPy 1.24+ 移除 np.bool 导致的 medpy 报错
if not hasattr(np, 'bool'):
    np.bool = bool

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

from medpy import metric
import torch.multiprocessing as multiprocessing
import pandas as pd  # 新增

# 从sam2导入必要的模块
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 导入骨架提取相关库
try:
    from skimage.morphology import skeletonize
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    print("警告: 缺少 scikit-image，无法使用骨架提取功能")


def safe_imread(file_path, grayscale=True):
    """安全读取图像文件，避免libpng警告"""
    try:
        if grayscale:
            # 使用PIL读取并转换为灰度图，避免ICC配置文件警告
            with Image.open(file_path) as img:
                if img.mode != 'L':
                    img = img.convert('L')
                return np.array(img)
        else:
            # 对于彩色图像，也使用PIL避免警告
            with Image.open(file_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return np.array(img)
    except Exception as e:
        print(f"警告: 无法读取文件 {file_path}: {e}")
        return None


## 辅助函数
def find_bounding_box(mask):
    assert mask.ndim == 3, "The mask must be a 3D numpy array."
    ones_indices = np.where(mask == 1)
    if not ones_indices[0].size:
        return None
    min_z = np.min(ones_indices[0])
    max_z = np.max(ones_indices[0])
    min_y = np.min(ones_indices[1])
    max_y = np.max(ones_indices[1])
    min_x = np.min(ones_indices[2])
    max_x = np.max(ones_indices[2])

    return min_z, max_z, min_y, max_y, min_x, max_x

def find_bounding_box_2d(mask):
    """
    在2D二值掩码中查找边界框
    
    Args:
        mask: 2D二值掩码
        
    Returns:
        min_y, max_y, min_x, max_x: 边界框坐标，如果没有前景区域则返回None
    """
    assert mask.ndim == 2, "掩码必须是2D数组"
    ones_indices = np.where(mask > 0)
    
    if not ones_indices[0].size:
        return None, None, None, None
        
    min_y = np.min(ones_indices[0])
    max_y = np.max(ones_indices[0])
    min_x = np.min(ones_indices[1])
    max_x = np.max(ones_indices[1])
    
    return min_y, max_y, min_x, max_x

def show_mask(mask, ax, random_color=False):
    """
    在matplotlib轴上可视化掩码
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    """
    在matplotlib轴上可视化边界框
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_points(coords, labels, ax, marker_size=375):
    """
    在matplotlib轴上可视化点提示
    """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def extract_foreground_points(cam, img, num_points=5):
    """
    从CAM和图像中提取前景点
    
    Args:
        cam: CAM掩码
        img: 原始图像
        num_points: 提取的前景点数量
        
    Returns:
        foreground_points: 前景点列表
    """
    # 归一化图像
    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
    
    # 使用CAM掩码获取高响应区域
    masked_img = cam * img_normalized
    
    if masked_img.sum() == 0:
        return []
    
    # 找到高强度像素位置
    hist, bins = np.histogram(masked_img, bins=10, range=[np.min(masked_img), np.max(masked_img)])
    high_intensity_threshold = bins[-3]  # 使用较高的阈值
    foreground_indices = np.where(masked_img >= high_intensity_threshold)
    
    if len(foreground_indices[0]) == 0:
        return []
        
    # 提取均匀分布的点
    indices = np.linspace(0, len(foreground_indices[0]) - 1, num_points).astype(int)
    
    points = []
    for idx in indices:
        points.append([foreground_indices[1][idx], foreground_indices[0][idx]])
    
    return points

def extract_foreground_points_from_quad_mask(quad_mask, num_points=6):
    """
    从四值掩码中提取前景点，按优先级采样：
    1. 优先从高置信区域(255)采样
    2. 如果不够，从中置信区域(128)补充
    3. 如果仍然不够，从低置信区域(64)补充
    
    Args:
        quad_mask: 四值掩码 (0=背景, 64=低置信, 128=中置信, 255=高置信)
        num_points: 提取的前景点数量，默认6个
        
    Returns:
        foreground_points: 前景点列表
    """
    points = []
    
    # 第一步：优先从高置信区域(255)选取
    high_conf_indices = np.where(quad_mask == 255)
    if len(high_conf_indices[0]) > 0:
        num_from_high = min(num_points, len(high_conf_indices[0]))
        np.random.seed(g_args.seed)
        random_indices = np.random.choice(len(high_conf_indices[0]), size=num_from_high, replace=False)
        
        for idx in random_indices:
            points.append([high_conf_indices[1][idx], high_conf_indices[0][idx]])
    
    # 如果已经够了，直接返回
    if len(points) >= num_points:
        return points[:num_points]
    
    # 第二步：从中置信区域(128)补充
    remaining_points = num_points - len(points)
    if remaining_points > 0:
        mid_conf_indices = np.where(quad_mask == 128)
        if len(mid_conf_indices[0]) > 0:
            num_from_mid = min(remaining_points, len(mid_conf_indices[0]))
            np.random.seed(g_args.seed)
            random_indices = np.random.choice(len(mid_conf_indices[0]), size=num_from_mid, replace=False)
            
            for idx in random_indices:
                points.append([mid_conf_indices[1][idx], mid_conf_indices[0][idx]])
    
    # 如果已经够了，直接返回
    if len(points) >= num_points:
        return points[:num_points]
    
    # 第三步：从低置信区域(64)补充
    remaining_points = num_points - len(points)
    if remaining_points > 0:
        low_conf_indices = np.where(quad_mask == 64)
        if len(low_conf_indices[0]) > 0:
            num_from_low = min(remaining_points, len(low_conf_indices[0]))
            np.random.seed(g_args.seed)
            random_indices = np.random.choice(len(low_conf_indices[0]), size=num_from_low, replace=False)
            
            for idx in random_indices:
                points.append([low_conf_indices[1][idx], low_conf_indices[0][idx]])
    
    # 返回收集到的所有点（可能少于num_points）
    return points

def generate_combinations(points):
    """
    生成前景点组合，总共41组提示：
    - 单点组合：所有单点 (6个)
    - 双点组合：所有两点组合 (15个)
    - 三点组合：所有三点组合 (20个)
    - 总计：6 + 15 + 20 = 41个组合
    """
    from itertools import combinations
    num_points = len(points)
    if num_points == 0:
        return []

    all_combinations = []
    
    # 单点组合：6个
    single_combinations = [(p,) for p in points]
    all_combinations.extend(single_combinations)
    
    # 双点组合：C(6,2) = 15个
    if num_points >= 2:
        pair_combinations = list(combinations(points, 2))
        all_combinations.extend(pair_combinations)
    
    # 三点组合：C(6,3) = 20个
    if num_points >= 3:
        triple_combinations = list(combinations(points, 3))
        all_combinations.extend(triple_combinations)
    
    # 验证组合数量
    expected_count = num_points + (num_points * (num_points - 1) // 2) + (num_points * (num_points - 1) * (num_points - 2) // 6)
    actual_count = len(all_combinations)
    
    if actual_count != expected_count:
        print(f"警告: 预期生成 {expected_count} 个组合，实际生成 {actual_count} 个")
    
    return all_combinations

def get_largest_connected_component(binary_mask):
    """
    获取二值掩码中最大的连通域
    
    Args:
        binary_mask: 二值掩码 (0/255 或 0/1)
        
    Returns:
        largest_cc: 最大连通域的掩码 (0/255)
    """
    # 确保是二值图像
    binary_mask = (binary_mask > 127).astype(np.uint8) * 255
    
    # 使用cv2查找连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    if num_labels <= 1:  # 没有前景区域
        return None
    
    # 找到面积最大的连通域（排除背景，索引0是背景）
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_cc = (labels == largest_label).astype(np.uint8) * 255
    
    return largest_cc

def extract_skeleton_from_mask(binary_mask):
    """
    从二值掩码的最大连通域中提取骨架曲线
    
    Args:
        binary_mask: 二值掩码 (0/255)
        
    Returns:
        skeleton: 骨架曲线图像 (0/255)，如果失败则返回None
        skeleton_length: 骨架长度（像素数）
        area: 掩码面积（像素数）
        slenderness: 长宽比 (skeleton_length^2 / area)
    """
    if not _HAS_SKIMAGE:
        return None, 0, 0, 0.0
    
    # 获取最大连通域
    largest_cc = get_largest_connected_component(binary_mask)
    if largest_cc is None:
        return None, 0, 0, 0.0
    
    try:
        # 提取骨架
        skeleton = skeletonize(largest_cc.astype(bool))
        skeleton = skeleton.astype(np.uint8) * 255
        
        # 计算骨架长度和面积
        skeleton_length = int(np.sum(skeleton > 0))
        area = int(np.sum(largest_cc > 0))
        
        # 计算长宽比
        slenderness = float(skeleton_length * skeleton_length / area) if area > 0 else 0.0
        
        return skeleton, skeleton_length, area, slenderness
    except Exception as e:
        print(f"骨架提取失败: {e}")
        return None, 0, 0, 0.0

# 背景点最远点采样（FPS）
def fps_on_mask(bg_mask, num_points=13, seed=42, max_candidates=5000, subsample_stride=None):
    rs = np.random.RandomState(seed)
    ys, xs = np.where(bg_mask)
    if len(xs) == 0:
        return []

    if subsample_stride is not None and subsample_stride > 1:
        ys = ys[::subsample_stride]
        xs = xs[::subsample_stride]

    N = len(xs)
    if N > max_candidates:
        idx = rs.choice(N, size=max_candidates, replace=False)
        xs, ys = xs[idx], ys[idx]

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    M = len(pts)
    k = min(num_points, M)
    if k == 0:
        return []

    first = rs.randint(M)
    selected_idx = [first]

    dmin = np.full(M, np.inf, dtype=np.float32)
    diff = pts - pts[first]
    dmin = np.minimum(dmin, np.sum(diff * diff, axis=1))

    for _ in range(1, k):
        nxt = int(np.argmax(dmin))
        selected_idx.append(nxt)
        diff = pts - pts[nxt]
        dmin = np.minimum(dmin, np.sum(diff * diff, axis=1))

    return [[int(pts[i, 0]), int(pts[i, 1])] for i in selected_idx]

def is_mask_valid(sam_mask, cam_trimap, img_gray, 
                  containment_threshold=0.95,
                  min_area_ratio_cam=300,
                  min_area_ratio_img=600,
                  max_area_ratio_cam=2,
                  max_connected_components=5,
                  skeleton_length_ratio=3,
                  slenderness_ratio=2):
    """
    判断掩码是否有效，基于新的筛选原则：
    1) SAM2掩码前景区域与（CAM掩码前景区域（255）+不确定区域（128））的交集/SAM2掩码 > containment_threshold
    2) 面积必须大于max(CAM掩码面积/min_area_ratio_cam, 图像总像素/min_area_ratio_img)且小于CAM掩码面积/max_area_ratio_cam
    3) CAM区域内候选掩码的连通域数量不得多于max_connected_components个
    4) 骨架曲线长度 >= CAM掩码骨架曲线长度的1/skeleton_length_ratio
    5) 长宽比 >= CAM掩码长宽比的slenderness_ratio倍
    
    Args:
        sam_mask: SAM2分割掩码
        cam_trimap: CAM三值掩码 (0=不响应, 128=不确定, 255=前景)
        img_gray: 灰度图像（用于获取图像总像素数）
        containment_threshold: 包含率阈值，默认0.95
        min_area_ratio_cam: CAM面积最小比例，默认300（即CAM面积/300）
        min_area_ratio_img: 图像总像素最小比例，默认600（即图像总像素/600）
        max_area_ratio_cam: CAM面积最大比例，默认2（即CAM面积/2）
        max_connected_components: 最大连通域数量，默认5
        skeleton_length_ratio: 骨架长度比例，默认3（即CAM骨架长度/3）
        slenderness_ratio: 长宽比比例，默认2（即CAM长宽比*2）
        
    Returns:
        is_valid: 是否有效
        metrics: 评估指标字典
    """
    if cam_trimap is None:
        return False, {}
    
    # 检查CAM掩码前景区域（255）
    cam_foreground_uncertain = ((cam_trimap == 255) | (cam_trimap == 128)).astype(np.uint8)
    cam_foreground_area = np.sum(cam_trimap == 255)
    
    # 如果CAM掩码前景区域为空集，则不输出任何候选掩码
    if cam_foreground_area == 0:
        return False, {'reason': 'CAM掩码前景区域为空'}
    
    # SAM2掩码二值化
    sam_bin = (sam_mask > 0).astype(np.uint8)
    sam_area = np.sum(sam_bin)
    
    if sam_area == 0:
        return False, {'reason': 'SAM2掩码为空'}
    
    # 第一步：计算SAM2掩码前景区域与（CAM前景+不确定区域）的交集比例
    intersection = np.sum(sam_bin * cam_foreground_uncertain)
    containment_ratio = intersection / sam_area if sam_area > 0 else 0.0
    
    if containment_ratio <= containment_threshold:
        return False, {
            'containment_ratio': containment_ratio,
            'reason': f'包含率 {containment_ratio:.3f} <= {containment_threshold}'
        }
    
    # 第二步：计算面积要求
    img_total_pixels = img_gray.shape[0] * img_gray.shape[1]
    cam_area = int(np.sum(cam_trimap == 255))
    min_area = max(cam_area / min_area_ratio_cam, img_total_pixels / min_area_ratio_img)
    max_area = cam_area / max_area_ratio_cam
    
    if sam_area < min_area or sam_area > max_area:
        return False, {
            'containment_ratio': containment_ratio,
            'sam_area': sam_area,
            'min_area': min_area,
            'max_area': max_area,
            'reason': f'面积 {sam_area} 不在范围 [{min_area:.1f}, {max_area:.1f}] 内'
        }
    
    # 第三步：检查CAM区域内候选掩码的连通域数量
    # 计算SAM2掩码与CAM区域（前景+不确定区域）的交集
    sam_in_cam_region = sam_bin * cam_foreground_uncertain
    sam_in_cam_binary = (sam_in_cam_region > 0).astype(np.uint8) * 255
    
    # 统计连通域数量
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sam_in_cam_binary, connectivity=8)
    num_connected_components = num_labels - 1  # 减去背景（索引0）
    
    if num_connected_components > max_connected_components:
        return False, {
            'containment_ratio': containment_ratio,
            'sam_area': sam_area,
            'num_connected_components': num_connected_components,
            'reason': f'CAM区域内连通域数量 {num_connected_components} > {max_connected_components}'
        }
    
    # 第四步：计算骨架和长宽比
    # 计算CAM掩码的骨架和长宽比
    cam_mask_binary = (cam_trimap == 255).astype(np.uint8) * 255
    cam_skeleton, cam_skeleton_length, cam_skeleton_area, cam_slenderness = extract_skeleton_from_mask(cam_mask_binary)
    
    if cam_skeleton is None or cam_skeleton_length == 0:
        # 如果CAM掩码无法提取骨架，则只检查面积和连通域要求
        metrics = {
            'containment_ratio': containment_ratio,
            'sam_area': sam_area,
            'cam_area': cam_area,
            'min_area': min_area,
            'max_area': max_area,
            'num_connected_components': num_connected_components,
            'is_valid': True,
            'reason': 'CAM掩码无骨架，仅通过面积和连通域检验'
        }
        return True, metrics
    
    # 计算SAM2掩码的骨架和长宽比
    sam_mask_binary = sam_bin.astype(np.uint8) * 255
    sam_skeleton, sam_skeleton_length, sam_skeleton_area, sam_slenderness = extract_skeleton_from_mask(sam_mask_binary)
    
    if sam_skeleton is None or sam_skeleton_length == 0:
        return False, {
            'containment_ratio': containment_ratio,
            'sam_area': sam_area,
            'reason': 'SAM2掩码无法提取骨架'
        }
    
    # 检查骨架长度要求：SAM2骨架长度 >= CAM骨架长度的1/skeleton_length_ratio
    required_min_skeleton = cam_skeleton_length / skeleton_length_ratio
    if sam_skeleton_length < required_min_skeleton:
        return False, {
            'containment_ratio': containment_ratio,
            'sam_area': sam_area,
            'num_connected_components': num_connected_components,
            'sam_skeleton_length': sam_skeleton_length,
            'cam_skeleton_length': cam_skeleton_length,
            'required_min_skeleton': required_min_skeleton,
            'reason': f'SAM2骨架长度 {sam_skeleton_length} < CAM骨架长度的1/{skeleton_length_ratio} ({required_min_skeleton:.1f})'
        }
    
    # 检查长宽比要求：SAM2长宽比 >= CAM长宽比的slenderness_ratio倍
    required_min_slenderness = cam_slenderness * slenderness_ratio
    if sam_slenderness < required_min_slenderness:
        return False, {
            'containment_ratio': containment_ratio,
            'sam_area': sam_area,
            'num_connected_components': num_connected_components,
            'sam_slenderness': sam_slenderness,
            'cam_slenderness': cam_slenderness,
            'required_min_slenderness': required_min_slenderness,
            'reason': f'SAM2长宽比 {sam_slenderness:.3f} < CAM长宽比的{slenderness_ratio}倍 ({required_min_slenderness:.3f})'
        }
    
    # 所有条件都满足
    metrics = {
        'containment_ratio': containment_ratio,
        'sam_area': sam_area,
        'cam_area': cam_area,
        'min_area': min_area,
        'max_area': max_area,
        'num_connected_components': num_connected_components,
        'sam_skeleton_length': sam_skeleton_length,
        'cam_skeleton_length': cam_skeleton_length,
        'sam_slenderness': sam_slenderness,
        'cam_slenderness': cam_slenderness,
        'is_valid': True
    }
    
    return True, metrics

def generate_sam_mask(img_gray, img_color, cam_trimap, quad_mask, predictor, vis_dir=None, img_name=None):
    """
    使用SAM2模型生成分割掩码，结合CAM三值掩码和四值阈值掩码：
    1) 背景点在CAM三值掩码的0区域（不响应区域）中选取
    2) 框提示来自CAM三值掩码中255区域（前景区域）的最小外接矩形
    3) 前景点提示在四值阈值掩码的高置信区域(255)选取，如果没有则在中置信区域(128)选取
    4) 生成41组前景点提示组合，每张图像得到多个候选掩码
    5) 质量过滤：基于新的筛选原则（包含率、面积、骨架长度、长宽比）
    
    Args:
        img_gray: 输入的灰度图像 (用于SAM处理)
        img_color: 输入的彩色图像 (用于可视化)
        cam_trimap: CAM三值掩码 (0=不响应, 128=不确定, 255=前景)
        quad_mask: 四值阈值掩码 (0=背景, 64=低置信, 128=中置信, 255=高置信)
        predictor: SAM2预测器
        vis_dir: 可视化保存目录
        img_name: 图像名称
        
    Returns:
        mask: SAM2生成的分割掩码，如果没有有效掩码则返回None
    """
    # 检查输入掩码
    if cam_trimap is None or quad_mask is None:
        return None

    # 检查CAM前景区域是否存在
    cam_foreground = (cam_trimap == 255)
    if np.sum(cam_foreground) == 0:
        return None

    # 基于CAM前景区域获得最小外接框
    min_y, max_y, min_x, max_x = find_bounding_box_2d(cam_foreground)
    if min_y is None:
        return None
    
    input_box = np.array([min_x, min_y, max_x, max_y])
    
    # 图像预处理
    img_normalized = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img_rgb = np.stack([img_normalized] * 3, axis=-1)  # 转换为RGB
    
    # 设置SAM2输入图像，显式转换为 PIL 图像以绕过可能的 numpy/torch 兼容性问题
    # (解决 TypeError: expected np.ndarray (got numpy.ndarray) 错误)
    img_pil = Image.fromarray(img_rgb)
    predictor.set_image(img_pil)
    
    # 背景点池：在CAM的0区域进行FPS采样，提前采样13个背景点
    bg_mask = (cam_trimap == 0)
    background_points_pool = fps_on_mask(bg_mask, num_points=13, seed=g_args.seed, max_candidates=5000, subsample_stride=None)
    # 转为 ndarray 便于后续切片
    background_points_pool = np.array(background_points_pool, dtype=np.int32) if len(background_points_pool) > 0 else np.zeros((0, 2), dtype=np.int32)
    
    # 从四值掩码中提取前景点：优先高置信区域(255)，其次中置信区域(128)，最后低置信区域(64)
    fg_points = extract_foreground_points_from_quad_mask(quad_mask, num_points=6)
    
    # 生成41组提示组合
    if fg_points:
        fg_combinations = generate_combinations(fg_points)
    else:
        # 如果没有前景点，只使用背景点和框
        fg_combinations = []
    
    # 分阶段尝试（按指定顺序早停）：评估全部multimask并早停，控制评估预算<=200
    def clamp_box(box, width, height):
        x0, y0, x1, y1 = box
        x0 = max(0, min(x0, width - 1))
        x1 = max(0, min(x1, width - 1))
        y0 = max(0, min(y0, height - 1))
        y1 = max(0, min(y1, height - 1))
        # 保证正面积
        if x1 <= x0:
            x1 = min(width - 1, x0 + 1)
        if y1 <= y0:
            y1 = min(height - 1, y0 + 1)
        return np.array([x0, y0, x1, y1], dtype=box.dtype)

    def scale_box(box, scale, width, height):
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        w = (box[2] - box[0]) * scale
        h = (box[3] - box[1]) * scale
        x0 = int(round(cx - w / 2.0))
        y0 = int(round(cy - h / 2.0))
        x1 = int(round(cx + w / 2.0))
        y1 = int(round(cy + h / 2.0))
        return clamp_box(np.array([x0, y0, x1, y1], dtype=box.dtype), width, height)

    H, W = img_gray.shape[:2]
    tight_box = input_box.copy()
    expand_box = scale_box(tight_box, 1.08, W, H)  # +8%
    shrink_box = scale_box(tight_box, 0.95, W, H)  # -5%

    # 生成前景点的单点/双点/三点组合
    from itertools import combinations
    fg_single = [(p,) for p in fg_points] if fg_points else []
    fg_pair = list(combinations(fg_points, 2)) if len(fg_points) >= 2 else []
    fg_triple_all = list(combinations(fg_points, 3)) if len(fg_points) >= 3 else []
    fg_triple = fg_triple_all  # 本策略需要全量20组

    # 评估函数：一次提示，评估所有multimask输出，若命中返回信息
    def evaluate_prompt(cur_points, cur_labels, cur_box, prompt_type):
        nonlocal eval_count
        # 准备数据类型
        point_coords = None
        point_labels = None
        if cur_points is not None and len(cur_points) > 0:
            point_coords = np.array(cur_points)
            point_labels = np.array(cur_labels)
            if 'cuda' in predictor.model.device.type and next(predictor.model.parameters()).dtype == torch.float16:
                point_coords = point_coords.astype(np.float16)
        box_arr = cur_box.copy() if cur_box is not None else None
        if box_arr is not None and 'cuda' in predictor.model.device.type and next(predictor.model.parameters()).dtype == torch.float16:
            box_arr = box_arr.astype(np.float16)

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_arr[None, :] if box_arr is not None else None,
            multimask_output=True
        )

        # 遍历所有multimask候选
        best_local = None
        for mi in range(masks.shape[0]):
            eval_count += 1
            current_mask = masks[mi]
            is_valid, metrics = is_mask_valid(
                current_mask, cam_trimap, img_gray,
                containment_threshold=g_args.containment_threshold,
                min_area_ratio_cam=g_args.min_area_ratio_cam,
                min_area_ratio_img=g_args.min_area_ratio_img,
                max_area_ratio_cam=g_args.max_area_ratio_cam,
                max_connected_components=g_args.max_connected_components,
                skeleton_length_ratio=g_args.skeleton_length_ratio,
                slenderness_ratio=g_args.slenderness_ratio
            )
            if is_valid:
                best_local = {
                    'mask': current_mask,
                    'metrics': metrics,
                    'points': point_coords,
                    'labels': point_labels,
                    'box': box_arr,
                    'type': prompt_type
                }
                break
            if eval_count >= 200:
                break
        return best_local

    # 评估预算计数器
    eval_count = 0
    best_candidate = None

    # 工具：生成若干随机背景子集（从FPS池中选），每个子集大小在[min_k, max_k]之间
    def make_random_bg_subsets(num_sets, min_k, max_k):
        rs = np.random.RandomState(g_args.seed)
        subsets = []
        pool = background_points_pool
        if pool.shape[0] == 0:
            return subsets
        for _ in range(num_sets):
            k = rs.randint(min_k, max_k + 1)
            k = min(k, pool.shape[0])
            if k == 0:
                subsets.append([])
            else:
                sel_idx = rs.choice(pool.shape[0], size=k, replace=False)
                subsets.append(pool[sel_idx].tolist())
        return subsets

    # 固定10个背景点（若池不足则用全部）
    fixed_bg_10 = background_points_pool[:min(10, background_points_pool.shape[0])].tolist()

    # 阶段1：单前景点 + 背景点(8-13随机) + tight_box，共 6*6=36 组（交叉）
    if best_candidate is None and eval_count < 200:
        bg_sets_stage1 = make_random_bg_subsets(6, 8, 13)
        for fg in fg_single:
            for bi, bgset in enumerate(bg_sets_stage1):
                pts = list(bgset) + [list(fg[0])]
                lbs = [0] * len(bgset) + [1]
                res = evaluate_prompt(pts, lbs, tight_box, f'1_single+bg{bi+1}+tight')
                if res is not None or eval_count >= 200:
                    best_candidate = res
                    break
            if best_candidate is not None or eval_count >= 200:
                break

    # 阶段2：双前景点 + 10背景点 + tight_box，共 15 组（固定背景和框）
    if best_candidate is None and eval_count < 200 and len(fg_pair) > 0:
        for comb in fg_pair:
            pts = fixed_bg_10 + [list(comb[0]), list(comb[1])]
            lbs = [0] * len(fixed_bg_10) + [1, 1]
            res = evaluate_prompt(pts, lbs, tight_box, '2_pair+bg10+tight')
            if res is not None or eval_count >= 200:
                best_candidate = res
                break

    # 阶段3：三前景点 + 10背景点 + tight_box，共 20 组（固定背景和框）
    if best_candidate is None and eval_count < 200 and len(fg_triple) > 0:
        for comb in fg_triple:
            pts = fixed_bg_10 + [list(comb[0]), list(comb[1]), list(comb[2])]
            lbs = [0] * len(fixed_bg_10) + [1, 1, 1]
            res = evaluate_prompt(pts, lbs, tight_box, '3_triple+bg10+tight')
            if res is not None or eval_count >= 200:
                best_candidate = res
                break

    # 阶段4：tight_box + 背景点(5-10随机)，6 组
    if best_candidate is None and eval_count < 200:
        bg_sets_stage4 = make_random_bg_subsets(6, 5, 10)
        for bi, bgset in enumerate(bg_sets_stage4):
            pts = list(bgset)
            lbs = [0] * len(bgset)
            res = evaluate_prompt(pts, lbs, tight_box, f'4_bg{bi+1}+tight')
            if res is not None or eval_count >= 200:
                best_candidate = res
                break

    # 阶段5：shrink_box + 背景点(5-10随机)，6 组
    if best_candidate is None and eval_count < 200:
        bg_sets_stage5 = make_random_bg_subsets(6, 5, 10)
        for bi, bgset in enumerate(bg_sets_stage5):
            pts = list(bgset)
            lbs = [0] * len(bgset)
            res = evaluate_prompt(pts, lbs, shrink_box, f'5_bg{bi+1}+shrink')
            if res is not None or eval_count >= 200:
                best_candidate = res
                break

    # 阶段6：expand_box + 背景点(5-10随机)，6 组
    if best_candidate is None and eval_count < 200:
        bg_sets_stage6 = make_random_bg_subsets(6, 5, 10)
        for bi, bgset in enumerate(bg_sets_stage6):
            pts = list(bgset)
            lbs = [0] * len(bgset)
            res = evaluate_prompt(pts, lbs, expand_box, f'6_bg{bi+1}+expand')
            if res is not None or eval_count >= 200:
                best_candidate = res
                break

    # 阶段7：tight_box + 前景点（单点、双点、三点），共 6+15+20 组，固定框和 10 背景点
    if best_candidate is None and eval_count < 200:
        # 单点
        for comb in fg_single:
            pts = fixed_bg_10 + [list(comb[0])]
            lbs = [0] * len(fixed_bg_10) + [1]
            res = evaluate_prompt(pts, lbs, tight_box, '7_single+bg10+tight')
            if res is not None or eval_count >= 200:
                best_candidate = res
                break
        # 双点
        if best_candidate is None and eval_count < 200:
            for comb in fg_pair:
                pts = fixed_bg_10 + [list(comb[0]), list(comb[1])]
                lbs = [0] * len(fixed_bg_10) + [1, 1]
                res = evaluate_prompt(pts, lbs, tight_box, '7_pair+bg10+tight')
                if res is not None or eval_count >= 200:
                    best_candidate = res
                    break
        # 三点
        if best_candidate is None and eval_count < 200 and len(fg_triple) > 0:
            for comb in fg_triple:
                pts = fixed_bg_10 + [list(comb[0]), list(comb[1]), list(comb[2])]
                lbs = [0] * len(fixed_bg_10) + [1, 1, 1]
                res = evaluate_prompt(pts, lbs, tight_box, '7_triple+bg10+tight')
                if res is not None or eval_count >= 200:
                    best_candidate = res
                    break

    # 若找到候选则可视化并返回
    if best_candidate is not None:
        best_mask = best_candidate['mask']
        best_points = best_candidate['points']
        best_labels = best_candidate['labels']
        best_metrics = best_candidate['metrics']
        best_type = best_candidate['type']
        used_box = best_candidate['box'] if best_candidate['box'] is not None else tight_box

        if vis_dir is not None and img_name is not None:
            img_display = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            axs[0].imshow(img_display)
            axs[0].set_title('原始图像')
            axs[0].axis('off')
            cam_visual = np.zeros_like(img_rgb)
            cam_visual[cam_trimap == 255] = [255, 0, 0]
            cam_visual[cam_trimap == 128] = [0, 255, 0]
            axs[1].imshow(img_display)
            axs[1].imshow(cam_visual, alpha=0.5)
            axs[1].set_title('CAM三值掩码\n(红=前景, 绿=不确定)')
            axs[1].axis('off')
            quad_visual = np.zeros_like(img_rgb)
            quad_visual[quad_mask == 255] = [255, 0, 0]
            quad_visual[quad_mask == 128] = [255, 255, 0]
            quad_visual[quad_mask == 64] = [0, 255, 0]
            axs[2].imshow(img_display)
            axs[2].imshow(quad_visual, alpha=0.5)
            axs[2].set_title('四值阈值掩码\n(红=高置信, 黄=中置信, 绿=低置信)')
            axs[2].axis('off')
            axs[3].imshow(img_display)
            show_mask(best_mask, axs[3])
            show_box(used_box, axs[3])
            if best_points is not None and best_labels is not None and len(best_points) > 0:
                show_points(best_points, best_labels, axs[3])
            # 构建标题，使用新的指标
            title_parts = [f'SAM2分割结果', f'类型: {best_type}']
            if 'containment_ratio' in best_metrics:
                title_parts.append(f'包含率: {best_metrics["containment_ratio"]:.3f}')
            if 'sam_area' in best_metrics:
                title_parts.append(f'面积: {best_metrics["sam_area"]}')
            if 'sam_slenderness' in best_metrics:
                title_parts.append(f'长宽比: {best_metrics["sam_slenderness"]:.2f}')
            axs[3].set_title('\n'.join(title_parts))
            axs[3].axis('off')
            plt.suptitle(f"{img_name} - 分割可视化")
            plt.tight_layout()
            os.makedirs(vis_dir, exist_ok=True)
            plt.savefig(os.path.join(vis_dir, f"{img_name}_sam2_visualization.png"), bbox_inches='tight', pad_inches=0.1)
            plt.close()
        return best_mask.astype(np.uint8)

    # 无有效掩码
    return None

def calculate_metrics(pred, gt):
    """计算Dice, Precision, Recall, IoU"""
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        precision = metric.binary.precision(pred, gt)
        recall = metric.binary.recall(pred, gt)
        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt) - intersection
        iou = intersection / union
        return dice, precision, recall, iou
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 1.0, 1.0, 1.0
    else:
        return 0.0, 0.0, 0.0, 0.0

def find_gt_mask(image_path, gt_dir):
    """尝试在GT目录中找到对应的掩码"""
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
        gt_path = os.path.join(gt_dir, name_without_ext + ext)
        if os.path.exists(gt_path):
            return safe_imread(gt_path, grayscale=True)
    return None

# --- 多进程工作函数 ---

# 每个工作进程的全局变量
g_predictor = None
g_args = None

def init_worker(args):
    """每个工作进程的初始化函数"""
    global g_predictor, g_args
    g_args = args
    
    # 每个进程加载自己的模型副本
    # print(f"工作进程 {os.getpid()} 正在加载SAM2模型...")
    device = torch.device(args.device)
    
    # 尝试加载模型，处理 hydra 配置路径问题
    try:
        # 获取配置文件的绝对路径
        config_path = os.path.abspath(args.sam_config)
        checkpoint_path = os.path.abspath(args.sam_checkpoint)
        
        # 如果配置文件存在，尝试直接使用 build_sam2
        # SAM2 的 build_sam2 内部使用 hydra.compose，这在某些环境下可能需要初始化
        from sam2.build_sam import build_sam2
        
        # 尝试直接加载，如果失败则尝试手动加载配置
        try:
            sam = build_sam2(config_path, checkpoint_path, device=device)
        except Exception as e:
            # 手动加载配置并实例化，绕过 hydra 的搜索路径限制
            from omegaconf import OmegaConf
            from hydra.utils import instantiate
            
            cfg = OmegaConf.load(config_path)
            # 如果配置中有 _global_ 标记，omegaconf 可能已经处理了
            if "model" in cfg:
                model_cfg = cfg.model
            else:
                model_cfg = cfg
                
            sam = instantiate(model_cfg, _recursive_=True)
            
            # 加载权重
            if checkpoint_path is not None:
                sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
                sam.load_state_dict(sd)
            
            sam = sam.to(device)
            
        sam.eval()
        g_predictor = SAM2ImagePredictor(sam)
        
    except Exception as e:
        import traceback
        print(f"工作进程初始化失败: {e}")
        traceback.print_exc()
        raise e

def process_file_worker(cam_file):
    """
    处理单个CAM三值掩码文件的工作函数。
    它依赖于由init_worker初始化的g_predictor和g_args。
    """
    try:
        # 从全局参数获取路径
        image_dir = g_args.image_dir
        sam_label_dir = g_args.sam_output_dir
        quad_mask_dir = g_args.quad_mask_dir
        if g_args.visualize:
            vis_dir = os.path.join(sam_label_dir, "visualization")
        else:
            vis_dir = None

        # 获取文件名
        file_name = os.path.basename(cam_file)
        name_without_ext = os.path.splitext(file_name)[0]
        
        # 获取原始图像路径
        image_file = None
        for ext in [".jpg", ".png"]:
            path = os.path.join(image_dir, name_without_ext + ext)
            if os.path.exists(path):
                image_file = path
                break
        
        if image_file is None:
            return f"失败: {name_without_ext} - 找不到对应的图像文件"
        
        # 使用安全的图像读取函数
        img_color = safe_imread(image_file, grayscale=False)
        cam_trimap = safe_imread(cam_file, grayscale=True)
        
        if img_color is None or cam_trimap is None:
            return f"失败: {name_without_ext} - 无法读取图像或CAM三值掩码"
        
        # 转换BGR到RGB（如果需要）
        if len(img_color.shape) == 3 and img_color.shape[2] == 3:
            img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
        
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        # 调整CAM三值掩码大小以匹配图像
        if cam_trimap.shape != img_gray.shape:
            cam_trimap = cv2.resize(cam_trimap, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 读取四值阈值掩码
        quad_mask = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            cand_path = os.path.join(quad_mask_dir, f"{name_without_ext}{ext}")
            if os.path.exists(cand_path):
                quad_mask = safe_imread(cand_path, grayscale=True)
                break
        
        if quad_mask is None:
            return f"跳过: {name_without_ext} - 找不到对应的四值阈值掩码"
        
        # 调整四值掩码大小以匹配图像
        if quad_mask.shape != img_gray.shape:
            quad_mask = cv2.resize(quad_mask, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 使用 torch.no_grad() 来减少内存消耗并加速
        # 使用 autocast 进行混合精度推理以获得性能提升并避免类型错误
        with torch.no_grad():
            use_autocast = 'cuda' in g_args.device
            with torch.autocast(device_type=g_args.device.split(':')[0], dtype=torch.float16, enabled=use_autocast):
                mask = generate_sam_mask(img_gray, img_color, cam_trimap, quad_mask, g_predictor, 
                                        vis_dir, 
                                        name_without_ext if g_args.visualize else None)
        
        # 保存SAM2掩码，只有当mask不为None时才保存
        if mask is not None:
            mask_output_path = os.path.join(sam_label_dir, f"{name_without_ext}.png")
            cv2.imwrite(mask_output_path, mask * 255)
            return f"成功: {name_without_ext}"
        else:
            # 未生成有效掩码（可能因候选未通过有效性筛选）
            return f"跳过: {name_without_ext} - 无有效候选掩码"

    except Exception as e:
        # 在这里捕获并打印详细的追溯信息，以便更好地调试
        import traceback
        tb_str = traceback.format_exc()
        return f"失败: {os.path.basename(cam_file)} - 错误: {e}\nTraceback:\n{tb_str}"


def main(args):
    # 设置随机种子以保证可复现性
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed_all(args.seed)
        # 注意：设置cudnn.deterministic为True可能会影响性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 数据路径设置
    data_dir = args.data_dir
    # 允许通过参数覆盖默认路径
    cam_dir = args.cam_dir if getattr(args, 'cam_dir', None) is not None else os.path.join(data_dir, f"Layercam/{args.stage}/cam_trimap")
    image_dir = args.image_dir if getattr(args, 'image_dir', None) is not None else os.path.join(data_dir, "Train_image/")
    gt_dir = args.gt_dir if getattr(args, 'gt_dir', None) is not None else os.path.join(data_dir, "Train_label")  # 修改为Train_label目录
    quad_mask_dir = args.quad_mask_dir if getattr(args, 'quad_mask_dir', None) is not None else os.path.join(data_dir, "threshold_label")
    
    if not os.path.exists(cam_dir):
        print(f"警告: CAM目录 {cam_dir} 不存在")
        # 尝试使用aug目录作为替代
        alt_cam_dir = cam_dir.replace("raw", "aug") if "raw" in cam_dir else cam_dir
        if os.path.exists(alt_cam_dir):
            print(f"使用替代目录: {alt_cam_dir}")
            cam_dir = alt_cam_dir
        else:
            # 如果目录不存在，则创建它或使用上级目录
            os.makedirs(cam_dir, exist_ok=True)
    
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录 {image_dir} 不存在")
        exit(1)
    
    # 检查GT目录是否存在
    if not os.path.exists(gt_dir):
        print(f"警告: GT目录 {gt_dir} 不存在，将无法计算评估指标")
        print(f"可用的目录: {[d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]}")
    else:
        print(f"GT目录 {gt_dir} 存在，将计算评估指标")
    
    # 创建输出目录
    sam_label_dir = args.sam_output_dir if getattr(args, 'sam_output_dir', None) is not None else os.path.join(data_dir, "SAM_label")
    os.makedirs(sam_label_dir, exist_ok=True)
    
    if args.visualize:
        vis_dir = os.path.join(sam_label_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
    
    # 打印CAM全零掩码处理模式
    #if args.save_empty_cam:
        #print("模式: 将保存CAM全零掩码对应的分割结果（全零掩码）")
    #else:
       # print("模式: 将跳过CAM全零掩码，不保存对应的分割结果")
    
    # 将解析后的路径写回 args，供子进程使用
    args.image_dir = image_dir
    args.sam_output_dir = sam_label_dir
    args.quad_mask_dir = quad_mask_dir

    # 获取所有CAM文件
    cam_files = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
    print(f"找到 {len(cam_files)} 个CAM文件")
    if not cam_files:
        print("未找到任何CAM文件，退出。")
        return

    # 获取所有原始图像文件数量（.jpg和.png）
    image_files_jpg = glob.glob(os.path.join(image_dir, "*.jpg"))
    image_files_png = glob.glob(os.path.join(image_dir, "*.png"))
    all_image_files = image_files_jpg + image_files_png
    print(f"找到 {len(all_image_files)} 个原始图像文件")

    # --- 新增：评估指标统计 ---
    all_metrics = []
    # ---

    # 设置并启动多进程池
    ctx = multiprocessing.get_context("spawn")
    num_workers = min(ctx.cpu_count(), args.num_workers)
    print(f"使用 {num_workers} 个工作进程进行并行处理...")

    with ctx.Pool(processes=num_workers, initializer=init_worker, initargs=(args,)) as pool:
        results = list(tqdm(pool.imap_unordered(process_file_worker, cam_files), total=len(cam_files), desc="SAM2并行推理"))

    # 统计保存的掩码数量
    saved_masks = glob.glob(os.path.join(sam_label_dir, "*.png"))
    print(f"保存了 {len(saved_masks)} 个掩码文件")

    # --- 新增：对每个掩码进行评估 ---
    gt_found_count = 0
    for mask_path in saved_masks:
        filename = os.path.basename(mask_path)
        name_without_ext = os.path.splitext(filename)[0]
        # 找到原始图像路径
        image_file = None
        for ext in [".jpg", ".png"]:
            path = os.path.join(image_dir, name_without_ext + ext)
            if os.path.exists(path):
                image_file = path
                break
        if image_file is None:
            continue
        # 找到真实掩码
        gt_mask = find_gt_mask(image_file, gt_dir)
        if gt_mask is None:
            continue
        gt_found_count += 1
        # 读取预测掩码
        pred_mask = safe_imread(mask_path, grayscale=True)
        if pred_mask is None:
            continue
        # 与 calculate_intersection_and_metrics.py 保持一致的处理
        if pred_mask.shape != gt_mask.shape:
            print(f"警告: 预测掩码与 GT 尺寸不匹配，跳过 {filename}")
            continue

        pred_bin = (pred_mask > 0).astype(np.uint8)
        _, gt_bin = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)

        dice, precision, recall, iou = calculate_metrics(pred_bin, gt_bin)
        all_metrics.append({
            "filename": filename,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "iou": iou
        })
    
    print(f"找到 {gt_found_count} 个对应的GT文件，计算了 {len(all_metrics)} 个评估指标")
    # ---

    # 统计保存的分割可视化图数量
    if args.visualize:
        vis_dir = os.path.join(sam_label_dir, "visualization")
        vis_imgs = glob.glob(os.path.join(vis_dir, "*_sam2_visualization.png"))
        print(f"保存了 {len(vis_imgs)} 个分割可视化图")

    # 报告结果
    success_count = 0
    failures = []
    missing_image_count = 0  # 统计找不到对应图像的数量
    skipped_no_quad_mask_count = 0  # 统计找不到四值阈值掩码的数量
    skipped_no_valid_candidate_count = 0  # 统计无有效候选掩码的数量
    
    for r in results:
        if r.startswith("成功"):
            success_count += 1
        elif r.startswith("跳过"):
            if "无有效候选掩码" in r:
                skipped_no_valid_candidate_count += 1
            elif "找不到对应的四值阈值掩码" in r:
                skipped_no_quad_mask_count += 1
        else:
            if "找不到对应的图像文件" in r:
                missing_image_count += 1
            else:
                failures.append(r)
    
    print(f"\n处理完成。")
    print(f"成功处理: {success_count} 个")
    print(f"找不到四值阈值掩码: {skipped_no_quad_mask_count} 个")
    print(f"无有效候选掩码: {skipped_no_valid_candidate_count} 个")
    print(f"总计处理: {success_count + skipped_no_quad_mask_count + skipped_no_valid_candidate_count}/{len(cam_files)}")
    
    # 显示找不到对应图像的总数
    if missing_image_count > 0:
        print(f"找不到对应图像文件: {missing_image_count} 个")
    
    # 显示其他类型的失败
    if failures:
        print("以下文件处理失败:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("所有文件均处理成功！")
    
    print(f"\nSAM2推理完成，结果保存在 {sam_label_dir}")

    # --- 打印和保存评估指标 ---
    # 1) 无论是否存在真实掩码，都先把所有成功保存的预测掩码对应的原图路径写入 CSV
    iou_dict = {m['filename']: m['iou'] for m in all_metrics}  # filename -> iou（仅当存在 GT 时才有值）

    image_paths_all = []
    ious_all = []
    for mask_path in saved_masks:
        filename = os.path.basename(mask_path)
        name_wo_ext = os.path.splitext(filename)[0]

        # 查找同名原始图像（支持 .jpg/.png）
        img_file = None
        for ext in ['.jpg', '.png']:
            candidate = os.path.join(image_dir, name_wo_ext + ext)
            if os.path.exists(candidate):
                img_file = candidate
                break
        image_paths_all.append(img_file)
        # 若没有对应 GT，则 IOU 记为 NaN
        ious_all.append(iou_dict.get(filename, np.nan))

    mask_iou_df = pd.DataFrame({'image_pth': image_paths_all, 'iou': ious_all})
    splits_dir = os.path.join(data_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    saved_masks_csv = os.path.join(splits_dir, 'saved_masks.csv')
    mask_iou_df.to_csv(saved_masks_csv, index=False)
    print(f"掩码路径已保存到: {saved_masks_csv}")

    # 2) 如果确实计算到了评估指标，则生成额外的汇总报告
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)

        mean_dice = metrics_df['dice'].mean()
        std_dice = metrics_df['dice'].std()
        mean_precision = metrics_df['precision'].mean()
        std_precision = metrics_df['precision'].std()
        mean_recall = metrics_df['recall'].mean()
        std_recall = metrics_df['recall'].std()
        mean_iou = metrics_df['iou'].mean()
        std_iou = metrics_df['iou'].std()

        summary = pd.DataFrame([{
            "filename": "mean", "dice": mean_dice, "precision": mean_precision, "recall": mean_recall, "iou": mean_iou
        },{
            "filename": "std", "dice": std_dice, "precision": std_precision, "recall": std_recall, "iou": std_iou
        }])

        results_df = pd.concat([metrics_df, summary], ignore_index=True)
        results_csv_path = os.path.join(sam_label_dir, 'results.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"详细指标已保存至: {os.path.abspath(results_csv_path)}")

        print("\n------ 结果摘要 ------")
        print(f"平均 Dice    : {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"平均 IoU     : {mean_iou:.4f} ± {std_iou:.4f}")
        print(f"平均 Precision: {mean_precision:.4f} ± {std_precision:.4f}")
        print(f"平均 Recall   : {mean_recall:.4f} ± {std_recall:.4f}")
        print("----------------------\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="使用SAM2对混凝土裂缝图像进行推理")
    # 使用示例:
    # python 5_sam2_inference1.py --data_dir ../data/ConcreteData --visualize  # 使用CAM三值掩码和四值阈值掩码引导SAM2分割，基于新筛选逻辑
    parser.add_argument('--data_dir', type=str, default='../data/ConcreteData', help='数据目录')
    parser.add_argument('--sam_checkpoint', type=str, default='../checkpoint/sam2/sam2_hiera_large.pt', help='SAM2模型检查点路径')
    parser.add_argument('--sam_config', type=str, default='../checkpoint/sam2/sam2_hiera_l.yaml', help='SAM2模型配置文件路径')
    parser.add_argument('--stage', type=str, default='aug', choices=['raw', 'aug', 'refined'], help='使用的CAM阶段')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备')
    parser.add_argument('--visualize', action='store_true', help='是否保存可视化结果')

    parser.add_argument('--seed', type=int, default=42, help='用于复现的随机种子')
    parser.add_argument('--num_workers', type=int, default=1, help='用于并行处理的工作进程数')
    # 可选路径参数（未设置则使用默认推断路径）
    parser.add_argument('--image_dir', type=str, default=None, help='原始图像目录，默认 data_dir/Train_image')
    parser.add_argument('--cam_dir', type=str, default=None, help='CAM目录，默认 data_dir/Layercam/{stage}/cam_post')
    parser.add_argument('--gt_dir', type=str, default=None, help='GT目录，默认 data_dir/Train_label')
    parser.add_argument('--sam_output_dir', type=str, default=None, help='SAM输出目录，默认 data_dir/SAM_label')
    parser.add_argument('--quad_mask_dir', type=str, default=None, help='四值阈值掩码目录，默认 data_dir/threshold_label')
    
    # 筛选条件参数
    parser.add_argument('--containment_threshold', type=float, default=0.95, 
                       help='包含率阈值，SAM2掩码与CAM区域的交集比例必须大于此值 (默认: 0.95)')
    parser.add_argument('--min_area_ratio_cam', type=float, default=300, 
                       help='CAM面积最小比例，最小面积 = max(CAM面积/此值, 图像总像素/min_area_ratio_img) (默认: 300)')
    parser.add_argument('--min_area_ratio_img', type=float, default=600, 
                       help='图像总像素最小比例，最小面积 = max(CAM面积/min_area_ratio_cam, 图像总像素/此值) (默认: 600)')
    parser.add_argument('--max_area_ratio_cam', type=float, default=2, 
                       help='CAM面积最大比例，最大面积 = CAM面积/此值 (默认: 2)')
    parser.add_argument('--max_connected_components', type=int, default=5, #先用5再用8
                       help='CAM区域内候选掩码的最大连通域数量 (默认: 5)')
    parser.add_argument('--skeleton_length_ratio', type=float, default=3, 
                       help='骨架长度比例，SAM2骨架长度必须 >= CAM骨架长度/此值 (默认: 3)')
    parser.add_argument('--slenderness_ratio', type=float, default=2, 
                       help='长宽比比例，SAM2长宽比必须 >= CAM长宽比*此值 (默认: 2)')
    
    args = parser.parse_args()
    main(args) 
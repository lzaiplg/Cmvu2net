import os
import torch
import clip
from tqdm import tqdm
import glob
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import argparse
import cv2

def classify_thresh(probs, threshold=0.6):
    max_prob = np.max(probs)
    pred_cls = np.argmax(probs)
    if max_prob >= threshold:
        return pred_cls, True
    else:
        return -1, False

def pr_result(cm, label_gt, label_pred):
    print(cm)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    print(f'TN:{TN}, TP:{TP}, FN:{FN}, FP:{FP}')
    unique_gt = set(label_gt)
    unique_pred = set(label_pred)
    if unique_gt == {0, 1} and unique_pred == {0, 1}:
        val_cls_acc = accuracy_score(label_gt, label_pred)
        val_cls_prec = precision_score(label_gt, label_pred, zero_division=0, average='binary', pos_label=1)
        val_cls_recall = recall_score(label_gt, label_pred, zero_division=0, average='binary', pos_label=1)
        val_cls_f1 = f1_score(label_gt, label_pred, zero_division=0, average='binary', pos_label=1)
        print(f'{val_cls_acc}|{val_cls_prec}|{val_cls_recall}|{val_cls_f1}')
    else:
        print("警告：预测或标签中只有一种类别，无法计算二分类指标。")
        val_cls_acc = accuracy_score(label_gt, label_pred)
        print(f'准确率: {val_cls_acc}')

def get_clip_label(filenames, model, preprocess, save_path, data_dir="data/ConcreteData", device="cpu"):
    label_pred, label_gt = [], []
    prompts_no_crack = ['a photo of a clean concrete surface']
    prompts_has_crack = ['a photo of  a concrete surface with cracks']
    text = clip.tokenize(prompts_no_crack + prompts_has_crack).to(device)
    os.makedirs(os.path.join(save_path, "crack"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "no_crack"), exist_ok=True)
    result_df = pd.DataFrame(columns=["filename", "pred_label", "gt_label", "confidence"])
    results = []
    crack_count = 0
    no_crack_count = 0
    mode = 2
    threshold = 0.8
    for idx in tqdm(range(len(filenames)), ncols=70):  
        filename = filenames[idx]
        image = cv2.imread(filename)
        if image is None:
            print(f"无法读取图像: {filename}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        base_name = os.path.basename(filename)
        if base_name.startswith("crack_"):
            label_cls = 1
        else:
            label_cls = 0
        image_pil = Image.fromarray(image)
        image_preprocessed = preprocess(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image_preprocessed, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            max_prob = np.max(probs)
            pred_cls = np.argmax(probs)
        num_no_crack_prompts = len(prompts_no_crack)
        if mode == 1:
            is_crack_pred = pred_cls >= num_no_crack_prompts
            if max_prob >= threshold and is_crack_pred:
                pred = 1
            else:
                pred = 0
        elif mode == 2:
            pred = 1 if pred_cls >= num_no_crack_prompts else 0
        else:
            raise ValueError("mode只能为1或2")
        confidence = float(probs[0][pred_cls])
        label_pred.append(pred)
        label_gt.append(label_cls)
        results.append({
            "filename": base_name,
            "pred_label": pred,
            "gt_label": label_cls,
            "confidence": confidence
        })
        if pred == 1:
            save_folder = os.path.join(save_path, 'crack')
            crack_count += 1
        else:
            save_folder = os.path.join(save_path, 'no_crack')
            no_crack_count += 1
        image_resized = cv2.resize(image, (384, 384))
        save_filename = f"{os.path.splitext(base_name)[0]}_pred_{pred}.jpg"
        cv2.imwrite(os.path.join(save_folder, save_filename), cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(save_path, "clip_predictions.csv"), index=False)
    if label_pred and label_gt:
        cm = confusion_matrix(label_gt, label_pred, labels=[0, 1])
        pr_result(cm, label_gt, label_pred)
    else:
        print("没有足够的预测结果来计算混淆矩阵")
    print(f"CLIP预测为裂缝(crack)的图片数量: {crack_count}")
    print(f"CLIP预测为非裂缝(no_crack)的图片数量: {no_crack_count}")
    total_crack = sum([1 for gt in label_gt if gt == 1])
    total_no_crack = sum([1 for gt in label_gt if gt == 0])
    print(f"裂缝图像总数: {total_crack}")
    print(f"无裂缝图像总数: {total_no_crack}")
    if 'cm' in locals():
        crack_as_no_crack = cm[1, 0]
        no_crack_as_crack = cm[0, 1]
    else:
        crack_as_no_crack = None
        no_crack_as_crack = None
    print(f"裂缝图像被识别为无裂缝的数量: {crack_as_no_crack}")
    print(f"无裂缝图像被识别为有裂缝的数量: {no_crack_as_crack}")
    total_acc = 0.0
    if label_pred and label_gt:
        total_acc = accuracy_score(label_gt, label_pred)
        crack_idx = [i for i, gt in enumerate(label_gt) if gt == 1]
        if crack_idx:
            crack_acc = accuracy_score([label_gt[i] for i in crack_idx], [label_pred[i] for i in crack_idx])
        else:
            crack_acc = None
        no_crack_idx = [i for i, gt in enumerate(label_gt) if gt == 0]
        if no_crack_idx:
            no_crack_acc = accuracy_score([label_gt[i] for i in no_crack_idx], [label_pred[i] for i in no_crack_idx])
        else:
            no_crack_acc = None
        print(f"总准确率: {total_acc:.4f}")
        if crack_acc is not None:
            print(f"裂缝图像准确率: {crack_acc:.4f}")
        else:
            print("裂缝图像准确率: 无裂缝样本")
        if no_crack_acc is not None:
            print(f"无裂缝图像准确率: {no_crack_acc:.4f}")
        else:
            print("无无裂缝样本")
    return total_acc

def create_data_splits(data_dir="data/ConcreteData", output_dir="data/ConcreteData/splits", split_ratio=(0.8, 0.2)):
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(data_dir, "image")
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    image_files = sorted(image_files)
    if not image_files:
        print(f"在{image_dir}中未找到图像文件")
        return
    crack_images = [f for f in image_files if os.path.basename(f).startswith("crack_")]
    no_crack_images = [f for f in image_files if not os.path.basename(f).startswith("crack_")]
    np.random.seed(42)
    np.random.shuffle(crack_images)
    np.random.shuffle(no_crack_images)
    n_crack = len(crack_images)
    n_no_crack = len(no_crack_images)
    train_crack = crack_images[:int(n_crack * split_ratio[0])]
    valid_crack = crack_images[int(n_crack * split_ratio[0]):]
    train_no_crack = no_crack_images[:int(n_no_crack * split_ratio[0])]
    valid_no_crack = no_crack_images[int(n_no_crack * split_ratio[0]):]
    train_df = pd.DataFrame({"image_pth": train_crack + train_no_crack})
    valid_df = pd.DataFrame({"image_pth": valid_crack + valid_no_crack})
    train_df["label"] = [1] * len(train_crack) + [0] * len(train_no_crack)
    valid_df["label"] = [1] * len(valid_crack) + [0] * len(valid_no_crack)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(output_dir, "valid.csv"), index=False)
    print(f"数据集划分完成: 训练集 {len(train_df)}张, 验证集 {len(valid_df)}张")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用CLIP为混凝土裂缝图像生成弱标签")
    parser.add_argument('--data_dir', type=str, default="../data/ConcreteData", help='数据目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备')
    args = parser.parse_args()
    data_dir = args.data_dir
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告：未检测到可用的GPU，自动切换为CPU运行。")
        device = 'cpu'
    print(f"当前使用的设备: {device}")
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        exit(1)
    image_dir = os.path.join(data_dir, "image")
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录 {image_dir} 不存在")
        exit(1)
    splits_dir = os.path.join(data_dir, "splits")
    if not os.path.exists(splits_dir) or not os.listdir(splits_dir):
        print("创建数据集划分...")
        create_data_splits(data_dir=data_dir, output_dir=splits_dir)
    model_name = 'RN50x4'
    model_name_safe = model_name.replace('/', '_')
    print(f"\n--- 正在加载模型: {model_name} ---")
    try:
        model, preprocess = clip.load(model_name, device=device)
    except Exception as e:
        print(f"加载模型 {model_name} 失败: {e}")
        exit(1)
    valid_csv = os.path.join(splits_dir, "valid.csv")
    if not os.path.exists(valid_csv):
        print(f"验证集文件未找到: {valid_csv}")
        exit(1)
    valid_df = pd.read_csv(valid_csv)
    valid_files = valid_df["image_pth"].tolist()
    valid_save_path = os.path.join(data_dir, "CLIP_label", "valid")
    os.makedirs(valid_save_path, exist_ok=True)
    print(f"正在为验证集生成CLIP标签...")
    accuracy = get_clip_label(valid_files, model, preprocess, valid_save_path, data_dir=data_dir, device=device)
    print(f"模型 {model_name} 的验证集准确率: {accuracy:.4f}")
    train_csv = os.path.join(splits_dir, "train.csv")
    if os.path.exists(train_csv):
        train_df = pd.read_csv(train_csv)
        train_files = train_df["image_pth"].tolist()
        train_save_path = os.path.join(data_dir, "CLIP_label", "train")
        os.makedirs(train_save_path, exist_ok=True)
        print("正在为训练集生成CLIP标签...")
        get_clip_label(train_files, model, preprocess, train_save_path, data_dir=data_dir, device=device)
    print("\nCLIP标签生成过程结束!") 
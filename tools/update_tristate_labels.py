import argparse
import os
import sys
import csv
from typing import Tuple, List

import numpy as np
import cv2
from PIL import Image
import shutil

import warnings as _warnings
import os as _os
_os.environ.setdefault('PYTHONWARNINGS', 'ignore')
_warnings.filterwarnings('ignore')

BACKGROUND = 0
FOREGROUND = 1
UNCERTAIN = 255

EXTREMELY_HIGH = 255
HIGH = 204
VERY_LOW = 51

def read_mask(path: str) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
    unique_vals = np.unique(arr)
    is_four_value = any(v in unique_vals for v in [51, 204])
    
    if is_four_value:
        return arr
    else:
        out = np.zeros_like(arr, dtype=np.uint8)
        out[arr == 0] = 0
        out[arr == 255] = 1
        out[arr == 128] = 255
        return out

def save_mask(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    unique_vals = np.unique(arr)
    is_four_value = any(v in unique_vals for v in [51, 204])
    
    if is_four_value:
        Image.fromarray(arr.astype(np.uint8)).save(path)
    else:
        out = np.zeros_like(arr, dtype=np.uint8)
        out[arr == 0] = 0
        out[arr == 1] = 255
        out[arr == 255] = 128
        Image.fromarray(out).save(path)

def parse_train_list(list_path: str, data_root: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_rel, lbl_rel = line.split()[:2]
            img_path = img_rel if os.path.isabs(img_rel) else os.path.join(data_root, img_rel)
            lbl_path = lbl_rel if os.path.isabs(lbl_rel) else os.path.join(data_root, lbl_rel)
            pairs.append((img_path, lbl_path))
    return pairs

def update_single(mask: np.ndarray, prob_b: np.ndarray, hard_thresh: float, top_ratio: float) -> Tuple[np.ndarray, dict, np.ndarray]:
    h, w = mask.shape
    if prob_b.shape != mask.shape:
        h, w = mask.shape
        prob_b = cv2.resize(prob_b.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    new_mask = mask.copy()
    
    stats = {
        "num_uncertain": int((mask == UNCERTAIN).sum()),
        "num_foreground": int((mask == FOREGROUND).sum()),
        "num_background": int((mask == BACKGROUND).sum()),
        "uncertain_ratio": float((mask == UNCERTAIN).sum()) / mask.size,
        "ratio": float(top_ratio),
        "hard_thresh": float(hard_thresh),
        "num_bg_updated": 0,
        "num_fg_denoised": 0,
        "num_fg_unlocked": 0,
        "fg_unlock_reason": None,
        "stop_reason": None,
    }

    original_uncertain = (mask == UNCERTAIN)
    
    bg_unlock_candidates = original_uncertain & (prob_b >= 0.5)
    
    if bg_unlock_candidates.sum() > 0:
        bg_unlock_probs = prob_b[bg_unlock_candidates]
        k = int(np.ceil(len(bg_unlock_probs) * top_ratio))
        if k > 0:
            top_indices = np.argpartition(bg_unlock_probs, -k)[-k:]
            top_mask = np.zeros_like(bg_unlock_candidates, dtype=bool)
            top_coords = np.argwhere(bg_unlock_candidates)
            for idx in top_indices:
                y, x = top_coords[idx]
                top_mask[y, x] = True
            
            select = top_mask & (prob_b >= hard_thresh)
            if select.sum() > 0:
                new_mask[select] = BACKGROUND
                stats["num_bg_updated"] = int(select.sum())
    
    return new_mask, stats, bg_unlock_candidates

def update_six_value_single(mask: np.ndarray, prob_b: np.ndarray, hard_thresh: float, top_ratio: float,
                           denoise_top_ratio: float = 0.10,
                           denoise_trigger_unlock_ratio: float = 0.01,
                           enable_unlock: bool = True,
                           enable_denoise: bool = True) -> Tuple[np.ndarray, dict, np.ndarray]:
    h, w = mask.shape
    if prob_b.shape != mask.shape:
        h, w = mask.shape
        prob_b = cv2.resize(prob_b.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    
    new_mask = mask.copy()
    
    stats = {
        "num_very_low": int((mask == VERY_LOW).sum()),
        "num_extremely_high": int((mask == EXTREMELY_HIGH).sum()),
        "num_high": int((mask == HIGH).sum()),
        "num_background": int((mask == BACKGROUND).sum()),
        "very_low_ratio": float((mask == VERY_LOW).sum()) / mask.size,
        "ratio": float(top_ratio),
        "hard_thresh": float(hard_thresh),
        "num_bg_updated": 0,
        "num_fg_denoised": 0,
    }
    
    original_very_low = (mask == VERY_LOW)
    
    if enable_unlock and original_very_low.sum() > 0:
        very_low_probs = prob_b[original_very_low]
        k = int(np.ceil(len(very_low_probs) * top_ratio))
        
        if k > 0:
            top_indices = np.argpartition(very_low_probs, -k)[-k:]
            top_mask = np.zeros_like(original_very_low, dtype=bool)
            top_coords = np.argwhere(original_very_low)
            for idx in top_indices:
                y, x = top_coords[idx]
                top_mask[y, x] = True
            
            select = top_mask & (prob_b >= hard_thresh)
            if select.sum() > 0:
                new_mask[select] = BACKGROUND
                stats["num_bg_updated"] = int(select.sum())
    
    current_uncertain_ratio = float((new_mask == VERY_LOW).sum()) / new_mask.size if new_mask.size > 0 else 0.0
    
    if enable_denoise:
        fg_high = (new_mask == HIGH)
        candidates_mask = fg_high & (prob_b >= hard_thresh)
        num_candidates = int(candidates_mask.sum())
        if num_candidates > 0:
            cand_probs = prob_b[candidates_mask]
            k = int(np.ceil(num_candidates * float(denoise_top_ratio)))
            if k > 0:
                top_idx = np.argpartition(cand_probs, -k)[-k:]
                cand_coords = np.argwhere(candidates_mask)
                for idx in top_idx:
                    y, x = cand_coords[idx]
                    new_mask[y, x] = VERY_LOW
                stats["num_fg_denoised"] = int(k)
                current_uncertain_ratio = float((new_mask == VERY_LOW).sum()) / new_mask.size if new_mask.size > 0 else 0.0

    return new_mask, stats, original_very_low

def _rel_from_any_labels(lbl_path: str) -> str:
    parts = lbl_path.replace("\\", "/").split("/")
    if "labels" in parts[-2]:
        idx = len(parts) - 2
        return "/".join(parts[idx+1:]) if idx+1 < len(parts) else parts[-1]
    return os.path.basename(lbl_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Update tri-state labels by comprehensive dynamic rule")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--base_labels", required=True, help="Path to original labels directory")
    parser.add_argument("--prob_dir", required=True, help="Directory containing .npy background prob files")
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--round_index", type=int, required=True)
    
    parser.add_argument("--hard_thresh", type=float, default=0.8, help="Hard threshold for background probability")
    parser.add_argument("--top_ratio", type=float, default=None, help="Optional override; if None, use start_ratio + ratio_step*round")
    parser.add_argument("--start_ratio", type=float, default=0.1, help="Initial top ratio for VERY_LOW unlock")
    parser.add_argument("--ratio_step", type=float, default=0.1, help="Increment per round for VERY_LOW unlock")
    parser.add_argument("--max_ratio", type=float, default=0.8, help="Max ratio cap for VERY_LOW unlock")
    parser.add_argument("--six_unlock_cap", type=float, default=0.8, help="Additional hard cap for six-value VERY_LOW unlock")
    
    parser.add_argument("--denoise_top_ratio", type=float, default=0.10,
                        help="Top ratio among background-high candidates in 204 to downgrade to 51")
    parser.add_argument("--denoise_trigger_unlock_ratio", type=float, default=0.01,
                        help="Trigger downgrade when newly unlocked uncertain->background ratio falls below this value (e.g., 0.01)")
    
    args = parser.parse_args()

    if args.top_ratio is not None:
        round_ratio = float(args.top_ratio)
    else:
        start = float(args.start_ratio)
        step = float(args.ratio_step)
        maxr = float(args.max_ratio)
        cap = float(args.six_unlock_cap)
        round_ratio = start + step * int(args.round_index)
        round_ratio = min(round_ratio, maxr, cap)

    out_labels_dir = os.path.join(args.data_root, f"labels{args.round_index}")
    os.makedirs(out_labels_dir, exist_ok=True)

    prev_round_uncertain_ratio = None
    prev_prev_round_uncertain_ratio = None
    
    save_dir = os.path.dirname(args.prob_dir)
    
    if args.round_index > 0:
        prev_prob_dir = os.path.join(save_dir, f'prob_round{args.round_index - 1}')
        prev_ratio_file = os.path.join(prev_prob_dir, f"uncertain_ratio_round{args.round_index - 1}.txt")
        if os.path.isfile(prev_ratio_file):
            try:
                with open(prev_ratio_file, 'r', encoding='utf-8') as f:
                    prev_round_uncertain_ratio = float(f.read().strip())
            except Exception as e:
                print(f"警告: 无法读取上一轮不确定占比文件 {prev_ratio_file}: {e}")
        else:
            print(f"警告: 上一轮不确定占比文件不存在: {prev_ratio_file}")
    
    if args.round_index > 1:
        prev_prev_prob_dir = os.path.join(save_dir, f'prob_round{args.round_index - 2}')
        prev_prev_ratio_file = os.path.join(prev_prev_prob_dir, f"uncertain_ratio_round{args.round_index - 2}.txt")
        if os.path.isfile(prev_prev_ratio_file):
            try:
                with open(prev_prev_ratio_file, 'r', encoding='utf-8') as f:
                    prev_prev_round_uncertain_ratio = float(f.read().strip())
            except Exception as e:
                print(f"警告: 无法读取上上轮不确定占比文件 {prev_prev_ratio_file}: {e}")
        else:
            print(f"警告: 上上轮不确定占比文件不存在: {prev_prev_ratio_file}")
    
    need_denoise = False
    if prev_round_uncertain_ratio is not None and prev_prev_round_uncertain_ratio is not None:
        uncertain_ratio_decrease = prev_prev_round_uncertain_ratio - prev_round_uncertain_ratio
        need_denoise = uncertain_ratio_decrease < float(args.denoise_trigger_unlock_ratio)
        
        print(f"\n=== Round {args.round_index} 触发条件检查 ===")
        print(f"上上轮(Round {args.round_index - 2})不确定像素占比: {prev_prev_round_uncertain_ratio * 100:.4f}%")
        print(f"上一轮(Round {args.round_index - 1})不确定像素占比: {prev_round_uncertain_ratio * 100:.4f}%")
        print(f"不确定占比下降幅度: {uncertain_ratio_decrease * 100:.4f}%")
        print(f"触发阈值 (denoise_trigger_unlock_ratio): {float(args.denoise_trigger_unlock_ratio) * 100:.4f}%")
        
        if need_denoise:
            print(f"[触发] 前景降级策略")
            print(f"  原因: 不确定占比从 {prev_prev_round_uncertain_ratio * 100:.4f}% 变为 {prev_round_uncertain_ratio * 100:.4f}%")
            print(f"  差值 {uncertain_ratio_decrease * 100:.4f}% < 阈值 {float(args.denoise_trigger_unlock_ratio) * 100:.4f}%")
        else:
            print(f"[未触发]")
            print(f"  原因: 不确定占比从 {prev_prev_round_uncertain_ratio * 100:.4f}% 变为 {prev_round_uncertain_ratio * 100:.4f}%")
            print(f"  差值 {uncertain_ratio_decrease * 100:.4f}% >= 阈值 {float(args.denoise_trigger_unlock_ratio) * 100:.4f}%")
        print("=" * 50)
    elif args.round_index == 0:
        print(f"\n=== Round {args.round_index} 触发条件检查 ===")
        print("第一轮更新，无历史数据，不触发降级")
        print("=" * 50)
    elif args.round_index == 1:
        print(f"\n=== Round {args.round_index} 触发条件检查 ===")
        print("第二轮更新，只有一轮历史数据，不触发降级")
        print("=" * 50)

    sample_infos = []
    stats_rows = []
    pairs = parse_train_list(args.train_list, args.data_root)

    total_pixels = 0
    total_fg_uncertain = 0
    total_prev_uncertain_pixels = 0
    total_uncertain_after_all = 0
    any_ge_thresh = False
    has_four_value = False

    new_train_list_path = os.path.join(args.data_root, f"train_round{args.round_index}.txt")
    new_lines: List[str] = []

    for img_path, lbl_path in pairs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        prob_path = os.path.join(args.prob_dir, base + ".npy")
        if not os.path.isfile(prob_path):
            rel_from_labels = _rel_from_any_labels(lbl_path)
            out_path = os.path.join(args.data_root, f"labels{args.round_index}", rel_from_labels)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copy(lbl_path, out_path)
            out_lbl_rel = os.path.join(f"labels{args.round_index}", rel_from_labels).replace("\\", "/")
            try:
                img_rel = os.path.relpath(img_path, args.data_root).replace("\\", "/")
            except ValueError:
                img_rel = img_path.replace("\\", "/")
            new_lines.append(f"{img_rel} {out_lbl_rel}\n")
            continue

        mask = read_mask(lbl_path)
        prob_b = np.load(prob_path)
        if prob_b.shape != mask.shape:
            prob_b = cv2.resize(prob_b.astype(np.float32), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        unique_vals = np.unique(mask)
        is_four_value = any(v in unique_vals for v in [51, 204])
        
        mask_total_pixels = mask.size
        total_pixels += mask_total_pixels
        
        if is_four_value:
            has_four_value = True
            prev_uncertain = int((mask == VERY_LOW).sum())
            total_prev_uncertain_pixels += prev_uncertain
            new_mask, st, candidate = update_six_value_single(
                mask, prob_b,
                hard_thresh=args.hard_thresh,
                top_ratio=round_ratio,
                denoise_top_ratio=args.denoise_top_ratio,
                denoise_trigger_unlock_ratio=args.denoise_trigger_unlock_ratio,
                enable_unlock=True,
                enable_denoise=need_denoise
            )
            st["num_uncertain"] = st["num_very_low"]
            st["num_foreground"] = st["num_extremely_high"] + st["num_high"]
            st["uncertain_ratio"] = st["very_low_ratio"]
            if not need_denoise:
                st["num_fg_denoised"] = 0
            st["num_fg_unlocked"] = 0
            st["fg_unlock_reason"] = "four_value_mask_mode"
            st["stop_reason"] = None
            new_uncertain = int((new_mask == VERY_LOW).sum())
            total_uncertain_after_all += new_uncertain
            current_uncertain_for_info = new_uncertain
        else:
            prev_uncertain = int((mask == UNCERTAIN).sum())
            total_prev_uncertain_pixels += prev_uncertain
            new_mask, st, candidate = update_single(
                mask, prob_b, 
                hard_thresh=args.hard_thresh, 
                top_ratio=round_ratio
            )
            new_uncertain = int((new_mask == UNCERTAIN).sum())
            total_uncertain_after_all += new_uncertain
            current_uncertain_for_info = new_uncertain
        
        rel_from_labels = _rel_from_any_labels(lbl_path)
        base_lbl_path = os.path.join(args.base_labels, rel_from_labels)

        out_path = os.path.join(args.data_root, f"labels{args.round_index}", rel_from_labels)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_mask(out_path, new_mask)

        pred_label = (prob_b < 0.5).astype(np.uint8) * 255
        pred_out_path = os.path.join(args.data_root, f"labels{args.round_index}_pred", rel_from_labels)
        os.makedirs(os.path.dirname(pred_out_path), exist_ok=True)
        Image.fromarray(pred_label.astype(np.uint8)).save(pred_out_path)

        total_fg_uncertain += int(((mask == FOREGROUND) | (mask == UNCERTAIN)).sum())
        if candidate.sum() > 0:
            candidate_probs = prob_b[candidate]
            if (candidate_probs >= args.hard_thresh).any():
                any_ge_thresh = True

        out_lbl_rel = os.path.join(f"labels{args.round_index}", rel_from_labels).replace("\\", "/")
        try:
            img_rel = os.path.relpath(img_path, args.data_root).replace("\\", "/")
        except ValueError:
            img_rel = img_path.replace("\\", "/")
        new_lines.append(f"{img_rel} {out_lbl_rel}\n")
        
        sample_infos.append({
            "base": base,
            "is_four_value": is_four_value,
            "prob_path": prob_path,
            "original_mask_path": base_lbl_path,
            "out_rel": out_lbl_rel,
            "rel_from_labels": rel_from_labels,
            "stats": st,
            "mask_size": mask_total_pixels,
            "uncertain_after": current_uncertain_for_info
        })

    current_uncertain_ratio = 0.0
    if total_pixels > 0:
        current_uncertain_ratio = total_uncertain_after_all / total_pixels
    
    current_ratio_file = os.path.join(args.prob_dir, f"uncertain_ratio_round{args.round_index}.txt")
    os.makedirs(os.path.dirname(current_ratio_file), exist_ok=True)
    with open(current_ratio_file, 'w', encoding='utf-8') as f:
        f.write(f"{current_uncertain_ratio:.10f}")
    
    print(f"\n=== Round {args.round_index} 更新完成 ===")
    print(f"总像素数: {total_pixels:,}")
    print(f"上一轮不确定像素数: {total_prev_uncertain_pixels:,}")
    print(f"本轮更新后不确定像素数: {total_uncertain_after_all:,}")
    if total_pixels > 0:
        prev_ratio = total_prev_uncertain_pixels / total_pixels
        print(f"上一轮不确定像素占比: {prev_ratio * 100:.4f}%")
        print(f"本轮更新后不确定像素占比: {current_uncertain_ratio * 100:.4f}%")
        uncertain_ratio_decrease = prev_ratio - current_uncertain_ratio
        print(f"不确定占比变化: {uncertain_ratio_decrease * 100:.4f}%")
        
        if uncertain_ratio_decrease < float(args.denoise_trigger_unlock_ratio):
            print(f"\n[提示] 前景降级触发提示:")
            print(f"   不确定像素占比从 {prev_ratio * 100:.4f}% 变为 {current_uncertain_ratio * 100:.4f}%")
            print(f"   差值 {uncertain_ratio_decrease * 100:.4f}% < 阈值 {float(args.denoise_trigger_unlock_ratio) * 100:.4f}%")
            print(f"   -> 下一轮(Round {args.round_index + 1})将触发前景降级策略")
    print(f"本轮是否触发降级: {'是' if need_denoise else '否'}")
    print("=" * 50)

    for info in sample_infos:
        if info["is_four_value"]:
            info["stats"]["uncertain_ratio"] = (info["uncertain_after"] / info["mask_size"]) if info["mask_size"] > 0 else 0.0
        stats_rows.append([
            info["base"],
            round_ratio,
            info["stats"].get("num_bg_updated", 0),
            info["stats"].get("num_fg_denoised", 0),
            info["stats"].get("num_fg_unlocked", 0),
            info["stats"].get("uncertain_ratio", 0),
            info["stats"].get("fg_unlock_reason"),
            info["stats"].get("stop_reason")
        ])

    csv_path = os.path.join(args.prob_dir, f"update_round{args.round_index}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "ratio_used", "num_bg_updated", "num_fg_denoised", "num_fg_unlocked", 
                        "uncertain_ratio", "fg_unlock_reason", "stop_reason"])
        writer.writerows(stats_rows)

    with open(new_train_list_path, "w", encoding="utf-8") as fw:
        fw.writelines(new_lines)

    fg_uncertain_ratio = (total_fg_uncertain / max(1, total_pixels)) * 100.0
    should_stop = False
    stop_reasons = []
    if not any_ge_thresh:
        should_stop = True
        stop_reasons.append("no_background_prob_ge_hard_thresh")
    if fg_uncertain_ratio < 8.0:
        should_stop = True
        stop_reasons.append("fg_plus_uncertain_ratio_below_8pct")

    if should_stop:
        with open(os.path.join(args.prob_dir, "STOP"), "w", encoding="utf-8") as f:
            f.write("\n".join(stop_reasons))

    print("Update done. Stop:", should_stop, "; reasons:", stop_reasons)

if __name__ == "__main__":
    main()

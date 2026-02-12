import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import List, Optional

def run_cmd(cmd: List[str]) -> int:
    print("[run] ", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    proc.communicate()
    return proc.returncode

def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def discover_best_model_dir(save_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(save_dir, "best_recall"),
        os.path.join(save_dir, "best_model"), 
        os.path.join(save_dir, "best")
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    return None

def discover_latest_iter_dir(save_dir: str) -> Optional[str]:
    if not os.path.isdir(save_dir):
        return None
    latest_n = -1
    latest_path = None
    for name in os.listdir(save_dir):
        full = os.path.join(save_dir, name)
        if not os.path.isdir(full):
            continue
        if name.startswith("iter_"):
            try:
                n = int(name.split("iter_")[-1])
            except Exception:
                continue
            if n > latest_n:
                latest_n = n
                latest_path = full
    return latest_path

def load_run_state(state_path: str) -> dict:
    if os.path.isfile(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "current_total_iters": 0,
        "round_index": 0,
        "current_train_list": None,
        "last_update_reason": None,
        "stopped": False,
        "stop_reason": None,
    }

def save_run_state(state_path: str, state: dict) -> None:
    ensure_dir(os.path.dirname(state_path))
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def make_round_train_list(base_train_list: str, data_root: str, round_index: int, out_list_path: str) -> None:
    labels_dirname = f"labels{round_index}"
    with open(base_train_list, "r", encoding="utf-8") as fr, open(out_list_path, "w", encoding="utf-8") as fw:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 1:
                img_rel = parts[0]
                lbl_rel = img_rel.replace("images", labels_dirname).rsplit(".", 1)[0] + ".png"
                fw.write(f"{img_rel} {lbl_rel}\n")
            else:
                img_rel, lbl_rel = parts[0], parts[1]
                lbl_rel_new = lbl_rel.replace("/labels/", f"/{labels_dirname}/").replace("\\labels\\", f"\\{labels_dirname}\\")
                for seg in ["labels1", "labels2", "labels3", "labels4", "labels5", "labels6", "labels7", "labels8", "labels9", "labels10"]:
                    lbl_rel_new = lbl_rel_new.replace(f"/{seg}/", f"/{labels_dirname}/").replace(f"\\{seg}\\", f"\\{labels_dirname}\\")
                fw.write(f"{img_rel} {lbl_rel_new}\n")

def _to_rel_train_list_path(train_list_arg: str, data_root: str) -> str:
    data_root_norm = os.path.normpath(os.path.abspath(data_root))
    cand = os.path.normpath(train_list_arg)
    if not os.path.isabs(cand):
        cand_abs = os.path.normpath(os.path.abspath(cand))
    else:
        cand_abs = cand
    try:
        rel = os.path.relpath(cand_abs, data_root_norm)
        if not rel.startswith('..'):
            return rel.replace('\\', '/')
    except ValueError:
        pass
    base = os.path.basename(train_list_arg)
    if base.lower().endswith('.txt'):
        return base
    return train_list_arg.replace('\\', '/')

def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic background update training controller")
    parser.add_argument("--config", required=True, help="Path to PaddleSeg config yml")
    parser.add_argument("--data_root", required=True, help="Dataset root, e.g., data/deepcrack448sam")
    parser.add_argument("--train_list", required=True, help="Initial train list file, relative to data_root or absolute")
    parser.add_argument("--save_dir", required=True, help="Training save_dir (same as PaddleSeg --save_dir)")
    parser.add_argument("--total_iters", type=int, default=10000)
    parser.add_argument("--stage_interval", type=int, default=100, help="Train stage size in iters between evaluations")
    parser.add_argument("--update_interval", type=int, default=1000, help="Iters between dynamic updates")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--python_bin", default=sys.executable, help="Python executable to run PaddleSeg tools")
    parser.add_argument("--run_state", default=None, help="Path to run_state.json (default under save_dir)")
    parser.add_argument("--predict_out", default=None, help="Directory to save probability maps (default under save_dir/prob_round{n})")

    parser.add_argument("--hard_thresh", type=float, help="Background hard threshold for update_tristate_labels.py")
    parser.add_argument("--top_ratio", type=float, help="Background top ratio for update_tristate_labels.py")
    parser.add_argument("--foreground_background_thresh", type=float, help="Foreground denoising background confidence threshold")
    parser.add_argument("--foreground_max_update_ratio", type=float, help="Foreground denoising max update ratio")
    parser.add_argument("--foreground_hard_thresh", type=float, help="Foreground unlocking hard threshold")
    parser.add_argument("--foreground_unlock_ratio", type=float, help="Foreground unlocking ratio")
    parser.add_argument("--uncertain_threshold", type=float, help="Uncertain region ratio threshold for foreground unlocking")

    parser.add_argument("--opts", nargs=argparse.REMAINDER, help="Extra PaddleSeg opts: key value ...")

    args = parser.parse_args()

    run_state_path = args.run_state or os.path.join(args.save_dir, "run_state.json")
    state = load_run_state(run_state_path)

    current_iters = int(state.get("current_total_iters", 0))
    round_index = int(state.get("round_index", 0))

    if state.get("current_train_list") is None:
        if os.path.isabs(args.train_list):
            state["current_train_list"] = args.train_list
        else:
            state["current_train_list"] = os.path.join(args.data_root, args.train_list)
        save_run_state(run_state_path, state)

    ensure_dir(args.save_dir)

    while current_iters < args.total_iters:
        next_target = min(args.total_iters, current_iters + args.stage_interval)

        train_cmd = [
            args.python_bin,
            os.path.join("PaddleSeg", "tools", "train.py"),
            "--config", args.config,
            "--do_eval",
            "--use_vdl",
            "--save_interval", "100",
            "--log_iters", "10",
            "--iters", str(next_target),
            "--device", args.device,
            "--batch_size", str(args.batch_size),
            "--save_dir", args.save_dir,
            "--opts",
            "train_dataset.dataset_root", args.data_root,
            "train_dataset.train_path", _to_rel_train_list_path(state["current_train_list"], args.data_root),
        ]
        latest_ckpt = discover_latest_iter_dir(args.save_dir)
        if latest_ckpt:
            train_cmd.extend(["--resume_model", latest_ckpt])
        if args.opts:
            train_cmd.extend(args.opts)
        code = run_cmd(train_cmd)
        if code != 0:
            print("Training failed.", file=sys.stderr)
            sys.exit(code)

        current_iters = next_target
        state["current_total_iters"] = current_iters
        save_run_state(run_state_path, state)

        if current_iters % args.update_interval == 0:
            round_index += 1
            state["round_index"] = round_index

            best_dir = discover_best_model_dir(args.save_dir)
            if best_dir is None:
                print("No best model directory found; attempting to use latest checkpoint.")
                best_dir = os.path.join(args.save_dir, f"iter_{current_iters}")

            prob_dir = args.predict_out or os.path.join(args.save_dir, f"prob_round{round_index}")
            ensure_dir(prob_dir)

            pred_cmd = [
                args.python_bin,
                os.path.join("tools", "predict_prob.py"),
                "--config", args.config,
                "--model_dir", best_dir,
                "--data_root", args.data_root,
                "--train_list", state["current_train_list"],
                "--out_dir", prob_dir,
                "--device", args.device,
            ]
            code = run_cmd(pred_cmd)
            if code != 0:
                print("Prediction failed.", file=sys.stderr)
                sys.exit(code)

            update_cmd = [
                args.python_bin,
                os.path.join("tools", "update_tristate_labels.py"),
                "--data_root", args.data_root,
                "--base_labels", os.path.join(args.data_root, "labels"),
                "--prob_dir", prob_dir,
                "--train_list", state["current_train_list"],
                "--round_index", str(round_index),
            ]
            if args.hard_thresh is not None:
                update_cmd.extend(["--hard_thresh", str(args.hard_thresh)])
            if args.top_ratio is not None:
                update_cmd.extend(["--top_ratio", str(args.top_ratio)])
            if args.foreground_background_thresh is not None:
                update_cmd.extend(["--foreground_background_thresh", str(args.foreground_background_thresh)])
            if args.foreground_max_update_ratio is not None:
                update_cmd.extend(["--foreground_max_update_ratio", str(args.foreground_max_update_ratio)])
            if args.foreground_hard_thresh is not None:
                update_cmd.extend(["--foreground_hard_thresh", str(args.foreground_hard_thresh)])
            if args.foreground_unlock_ratio is not None:
                update_cmd.extend(["--foreground_unlock_ratio", str(args.foreground_unlock_ratio)])
            if args.uncertain_threshold is not None:
                update_cmd.extend(["--uncertain_threshold", str(args.uncertain_threshold)])

            code = run_cmd(update_cmd)
            if code != 0:
                print("Label update failed.", file=sys.stderr)
                sys.exit(code)

            stop_flag = os.path.join(prob_dir, "STOP")
            if os.path.isfile(stop_flag):
                state["stopped"] = True
                state["stop_reason"] = "Stop conditions met (no >=0.98 candidates or <8% fg+uncertain)."
                save_run_state(run_state_path, state)
                print("Stop conditions met. Ending controller loop.")
                break

            round_train_list = os.path.join(args.data_root, f"train_round{round_index}.txt")
            make_round_train_list(state["current_train_list"], args.data_root, round_index, round_train_list)
            state["current_train_list"] = round_train_list
            save_run_state(run_state_path, state)

    print("Controller finished. Final state saved at:", run_state_path)

if __name__ == "__main__":
    main()

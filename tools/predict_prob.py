import argparse
import os
import sys
import numpy as np
from typing import List, Tuple
import os as _os
import warnings as _warnings
_os.environ.setdefault('PYTHONWARNINGS', 'ignore')
_warnings.filterwarnings('ignore')

def list_train_images(train_list_path: str, data_root: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(train_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 1:
                img_rel = parts[0]
                lbl_rel = img_rel.replace("images", "labels").rsplit(".", 1)[0] + ".png"
            else:
                img_rel, lbl_rel = parts[0], parts[1]
            img_path = img_rel if os.path.isabs(img_rel) else os.path.join(data_root, img_rel)
            pairs.append((img_path, lbl_rel))
    return pairs

def infer_background_prob(config_path: str, model_dir: str, image_path: str, device: str = "gpu") -> np.ndarray:
    raise NotImplementedError

def _strip_random_transforms(transform_list):
    clean = []
    for t in transform_list:
        name = getattr(t, "__class__", type(t)).__name__
        if name.startswith("Random"):
            continue
        clean.append(t)
    return clean

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict background probabilities for training set")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_dir", required=True, help="Directory that contains model.pdparams")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sys.path.insert(0, os.path.abspath("."))
    import paddle
    from PaddleSeg.paddleseg.cvlibs import Config
    from PaddleSeg.paddleseg.datasets.dataset import Dataset as PPDataset

    if args.device == "gpu":
        try:
            paddle.set_device('gpu:0')
            print("使用GPU进行预测")
        except Exception as e:
            print(f"GPU初始化失败: {e}")
            print("回退到CPU进行预测")
            paddle.set_device('cpu')
    else:
        paddle.set_device('cpu')
        print("使用CPU进行预测")

    modified_opts = [
        'train_dataset.dataset_root', args.data_root,
        'train_dataset.train_path', args.train_list,
        'val_dataset.dataset_root', args.data_root,
        'val_dataset.val_path', args.train_list 
    ]
    cfg = Config(args.config, opts=modified_opts)
    base_train = getattr(cfg, 'train_dataset', None)
    base_val = getattr(cfg, 'val_dataset', None)
    base_ds = base_val if base_val is not None else base_train
    if base_ds is None:
        raise RuntimeError("Cannot load base dataset from config")

    num_classes = getattr(base_ds, 'num_classes', 2)
    img_channels = getattr(base_ds, 'img_channels', 3)
    ignore_index = getattr(base_ds, 'ignore_index', 255)

    transforms = None
    if base_val is not None and getattr(base_val, 'transforms', None) is not None:
        transforms = getattr(base_val, 'transforms')
        transform_list = getattr(transforms, 'transforms', None) or transforms
    else:
        t = getattr(base_train, 'transforms', None)
        if t is None:
            raise RuntimeError("No transforms found in dataset config")
        t_list = getattr(t, 'transforms', None) or t
        transform_list = _strip_random_transforms(t_list)

    pp_dataset = PPDataset(
        mode='val',
        dataset_root=args.data_root,
        transforms=transform_list,
        num_classes=num_classes,
        img_channels=img_channels,
        val_path=args.train_list,
        ignore_index=ignore_index,
        edge=False,
    )

    pairs = list_train_images(args.train_list, args.data_root)
    base_names = [os.path.splitext(os.path.basename(p[0]))[0] for p in pairs]

    if len(base_names) != len(pp_dataset):
        print(f"[warn] mismatch between train_list lines ({len(base_names)}) and dataset size ({len(pp_dataset)});")

    model_path = os.path.join(args.model_dir, "model.pdparams")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = cfg.model
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    model.eval()

    dataset_batch_size = 50
    total_samples = len(pp_dataset)
    print(f"开始预测 {total_samples} 张图像，数据集批次大小: {dataset_batch_size}")
    
    for dataset_batch_start in range(0, total_samples, dataset_batch_size):
        dataset_batch_end = min(dataset_batch_start + dataset_batch_size, total_samples)
        print(f"处理数据集批次 {dataset_batch_start//dataset_batch_size + 1}/{(total_samples + dataset_batch_size - 1)//dataset_batch_size}: 图像 {dataset_batch_start+1}-{dataset_batch_end}")
        
        temp_train_list = os.path.join(args.out_dir, f"temp_train_batch_{dataset_batch_start}.txt")
        with open(temp_train_list, 'w') as f:
            for idx in range(dataset_batch_start, dataset_batch_end):
                if idx < len(base_names):
                    with open(args.train_list, 'r') as orig_f:
                        lines = orig_f.readlines()
                        if idx < len(lines):
                            f.write(lines[idx])
        
        temp_modified_opts = [
            'train_dataset.dataset_root', args.data_root,
            'train_dataset.train_path', temp_train_list,
            'val_dataset.dataset_root', args.data_root,
            'val_dataset.val_path', temp_train_list
        ]
        
        temp_cfg = Config(args.config, opts=temp_modified_opts)
        temp_dataset = PPDataset(
            mode='val',
            dataset_root=args.data_root,
            transforms=transform_list,
            num_classes=num_classes,
            img_channels=img_channels,
            val_path=temp_train_list,
            ignore_index=ignore_index,
            edge=False,
        )
        
        for local_idx in range(len(temp_dataset)):
            global_idx = dataset_batch_start + local_idx
            try:
                data = temp_dataset[local_idx]
                img = data['img']
                if not isinstance(img, paddle.Tensor):
                    img = paddle.to_tensor(img, dtype='float32')
                if img.ndim == 3:
                    img = img.unsqueeze(0)

                with paddle.no_grad():
                    logits = model(img)
                    if isinstance(logits, (list, tuple)):
                        logits = logits[0]
                    probs = paddle.nn.functional.softmax(logits, axis=1)
                    bg_prob = probs[0, 0].numpy().astype('float32')

                if global_idx < len(base_names):
                    base = base_names[global_idx]
                else:
                    base = f"sample_{global_idx:06d}"

                np.save(os.path.join(args.out_dir, base + ".npy"), bg_prob)
                
                del img, logits, probs, bg_prob
                
            except Exception as e:
                print(f"[warn] failed to infer index {global_idx}: {e}")
                continue
        
        if os.path.exists(temp_train_list):
            os.remove(temp_train_list)
        
        if args.device == "gpu":
            paddle.device.cuda.empty_cache()
        
        print(f"数据集批次 {dataset_batch_start//dataset_batch_size + 1} 完成")

    print("Prediction done.")

if __name__ == "__main__":
    main()

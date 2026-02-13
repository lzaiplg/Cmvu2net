# Cmvu2net

This repository contains the implementation of Cmvu2-net for crack segmentation.

## Environment Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training

To train the model, run the following command:

```bash
python pytorch/train.py --config configs/cmvu2net.yml --do_eval --use_vdl --save_interval 4 --log_iters 10 --epoch 800 --device gpu --num_workers 0 --save_dir ./output/cmvu2net_deepcrack --update_hard_thresh 0.9 --update_interval 16 --update_start_ratio 0.1 --update_ratio_step 0.1 --update_max_ratio 0.8 --denoise_top_ratio 0.1 --denoise_trigger_unlock_ratio 0.01 --opts train_dataset.dataset_root ./data/deepcrack train_dataset.train_path ./data/deepcrack/train.txt val_dataset.dataset_root ./data/deepcrack val_dataset.val_path ./data/deepcrack/val.txt
```

## Validation

To validate the model, run the following command:

```bash
python pytorch/val.py --config configs/cmvu2net.yml --model_path output\cmvu2net_deepcrack\model_iter_14288.pth --batch_size 16 --save_dir output\val_predictions\deepcrack_torch --device gpu --num_workers 0 --opts val_dataset.dataset_root .\data\deepcrack val_dataset.val_path .\data\deepcrack\val.txt
```

## Evaluation

To evaluate the model size and computational complexity (Parameters and FLOPs), run:

```bash
python tools/model_evaluation.py --config configs/cmvu2net.yml
```


## Datasets

1.DeepCrack：
```bash
Liu Y, Yao J, Lu X, et al. DeepCrack: A deep hierarchical feature learning architecture for crack segmentation[J]. Neurocomputing, 2019, 338: 139-153. https://doi.org/10.1016/j.neucom.2019.01.036.
```

2.Concrete3k：
```bash
Li Y, Ma R, Liu H, et al. Real-time high-resolution neural network with semantic guidance for crack segmentation[J]. Automation in Construction, 2023, 156: 105112. https://doi.org/10.1016/j.autcon.2023.105112.
CN：Yang 
```

3.FCN：
```bash
Li H, Yu Y, et al. Automatic pixel‐level crack detection and measurement using fully convolutional network[J]. Computer‐Aided Civil and Infrastructure Engineering, 2018, 33(12): 1090-1109. https://doi.org/10.1111/mice.12412.
```

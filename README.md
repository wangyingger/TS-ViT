# TS-ViT
# Code Running

## Requirements

python     >= 3.9

pytorch	>= 1.8.1

Apex (optional)

## Training

1. Put the pre-trained ViT model in `pretrained/`, and rename it to `ViT-B_16.npz`, you can download from [ViT pretrained](https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/ViT-B_16.npz).
2. Select a experiments setting file in `configs/`, and Modify the path of `dataset`.
3. Modify the path in `setup.py` in line 5, and you can change the log name and cuda visible by modify line 13,14.
4. Running the following code according to you pytorch version:

### Single GPU

```bash
python -m main.py
```

### Multiple GPUs

#### If pytorch < 1.12.0

```bash
python -m torch.distributed.launch --nproc_per_node 4 main.py 
```

#### If pytorch >= 1.12.0

```
torchrun --nproc_per_node 4 main.py
```

You need to change the number behind the `-nproc_per_node` to your number of GPUs

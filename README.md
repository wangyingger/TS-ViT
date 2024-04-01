# TS-ViT: Feature-Enhanced Transformer via Token Selection for Fine-Grained Image Recognition
# Code Running

## Requirements

python     >= 3.9

pytorch	>= 1.10.0

timm	== 0.9.12

## Training

1. Put the pre-trained ViT model in `pretrained/`, and rename it to `ViT-B_16.npz`, you can download from [ViT pretrained](https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/ViT-B_16.npz).
2. Select a experiments setting file in `configs/`, and Modify the path of `dataset`.
3. Running the following code according to you pytorch version:

### Single GPU

```bash
python -m main.py
```

### Multiple GPUs


```bash
python -m torch.distributed.launch --nproc_per_node 4 main.py 
```


You need to change the number behind the `-nproc_per_node` to your number of GPUs

# ColorPointTextureNet

## Install

## Train
```python
python main.py -w [writer_path] -o [output_path] -
```

#### Train VAE
```python
CUDA_VISIBLE_DEVICES=2 python main.py -w vae_2e-4_48_64 --lr 2e-4 --train-vae -emb 48
```

#### Train Main
```python
CUDA_VISIBLE_DEVICES=2 python main.py -w main_5e-4 --lr 5e-4 --train-main -emb 48
```

# ColorPointTextureNet

<<<<<<< Updated upstream
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
=======
## Train
#### Train VAE
```python
python main.py --train_vae -w [writer_path] -o [output_path] -emb [colour_emb_dim] --lr [learning_rate]
```

#### Train Main (Non-SampleNet)
```python
python main.py --train_main -w [writer_path] -o [output_path] --sampler [uniform|random|fps] -in [input_point_num] -out [sample_point_num] --vae_path [vae_model_path] --lr [learning_rate]
```

#### Train Main (SampleNet)
```python
python main.py --train-main -w [writer_path] -o [output_path] --sampler samplenet -in [input_point_num] -out [sample_point_num] --vae_path [vae_model_path] --lr [learning_rate] --lr_sam [samplenet_learning_rate]
```

## Test
#### Test VAE
```python
python main.py --test_vae -w [writer_path] -o [output_path] --vae_path [vae_model_path]
```

#### Test Main
```python
python main.py --test_main -w [writer_path] -o [output_path] --sampler [samplenet|uniform|random|fps]  -in [input_point_num] -out [sample_point_num] --model_path [model_path]
```

## Generation
#### Generation VAE
```python
python main.py --gen_vae -w [writer_path] -o [output_path] --vae_path [vae_model_path]
```

#### Generation Main
```python
python main.py --gen_main -w [writer_path] -o [output_path] --sampler [samplenet|uniform|random|fps]  -in [input_point_num] -out [sample_point_num] --model_path [model_path]
```

## Tensorboard
```python
tensorboard --logdir=runs
```
>>>>>>> Stashed changes

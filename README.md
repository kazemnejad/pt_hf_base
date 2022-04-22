# Base deep learning codebase for Huggingface projects

## Features
- Dependency injection using Jsonnet files
- Wandb
- Logging
- Mila/Slurm support
- HP tune

## Supported Models
- Encoder-Decoder (T5)
- Decoder Only (GPT-Neo)

## Supported Tasks
- Sequence to Sequence prediction

## How to use
**Train**
```shell
python src/main.py --configs "configs/t5a_debug.jsonnet" train
```
**Prediction**
```shell
python src/main.py --configs "configs/t5a_debug.jsonnet" predict
```

**Analyze**
```shell
python src/main.py --configs "configs/t5a_debug.jsonnet" analyze_all
```

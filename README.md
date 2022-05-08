# Research codebase for Huggingface-based projects

## Features
- Dependency injection using Jsonnet files
- Wandb integration
- Logging
- Mila/Slurm support
- HP tune
- Designed with reproducibility in mind

## Supported Models
- Encoder-Decoder (T5)
- Decoder Only (GPT-Neo)

## Supported Tasks
- Sequence to Sequence prediction

## How to use
Example usage involving **training**, **prediction**, **analysis**, **multiple seed**
```shell
#!/bin/bash

export APP_DS_SPLIT=simple
export WANDB_RUN_GROUP=example_exp

for SEED in `seq 1 3`; do
	export APP_DIRECTORY=experiments/example_exp
	export APP_EXPERIMENT_NAME=seed_$SEED
	export APP_SEED=$SEED
	export WANDB_JOB_TYPE=exp
	export WANDB_RUN_ID=random_id_seed_$SEED

	python src/main.py --configs 'configs/t5a_debug.jsonnet,configs/data/scan.jsonnet' \
	       train

	python src/main.py --configs 'configs/t5a_debug.jsonnet,configs/data/scan.jsonnet' \
	       predict --split test

	python src/main.py --configs 'configs/t5a_debug.jsonnet,configs/data/scan.jsonnet' \
	       combine_pred --split test

	python src/main.py --configs 'configs/t5a_debug.jsonnet,configs/data/scan.jsonnet' \
	       analyze_all

done

echo "Experiment finished!"
```

## Code structure

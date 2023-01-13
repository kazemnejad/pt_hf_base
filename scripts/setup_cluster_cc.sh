#!/bin/bash

VENV_PATH=~/venv_pt_hf_base

module load python/3.9
python -m venv $VENV_PATH
source $VENV_PATH/bin/activate
pip install pika wandb PyGithub InquirerPy

mkdir -p ~/scratch/containers
cd ~/scratch/containers/
module load singularity
singularity pull --arch amd64 library://kzmnjd/deeplr/pt:v7

module load gcc arrow scipy-stack
source $VENV_PATH/bin/activate
pip install torch torchvision transformers datasets sklearn sentencepiece seqeval
mkdir -p ~/scratch/experiments/hf_cache
mkdir -p ~/scratch/experiments/hf_ds_cache
mkdir -p ~/scratch/experiments/hf_module_cache
mkdir -p ~/scratch/experiments/wandb_cache_dir
export TRANSFORMERS_CACHE=~/scratch/experiments/hf_cache
export HF_DATASETS_CACHE=~/scratch/experiments/hf_ds_cache
export HF_MODULES_CACHE=~/scratch/experiments/hf_module_cache
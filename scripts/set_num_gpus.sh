#!/bin/bash

# If using SLURM env
if [ -z "$SLURM_JOB_ID" ]; then
  # Not running within a SLURM job
  export NUM_GPUS=$(nvidia-smi -L | wc -l)
else
  # Running within a SLURM job
  if [ -z "$SLURM_GPUS_ON_NODE" ]; then
    export NUM_GPUS=$(nvidia-smi -L | wc -l)
  else
    export NUM_GPUS=$SLURM_GPUS_ON_NODE
  fi
fi
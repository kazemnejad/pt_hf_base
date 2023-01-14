#!/bin/bash

# Assert $SWEEP_CONFIGS is set
if [ -z "$SWEEP_CONFIGS" ]; then
  echo "SWEEP_CONFIGS is not set"
  exit 1
fi

# Assert $SWEEP_NAME is set
if [ -z "$SWEEP_NAME" ]; then
  echo "SWEEP_NAME is not set"
  exit 1
fi

# Exit when any command fails
set -e

export APP_LAUNCHED_BY_MANUAL_SWEEPER=1
export APP_MANUAL_SWEEP=1

python scripts/manual_sweep.py \
  --sweep_name $SWEEP_NAME \
  --sweep_root_dir $SWEEP_ROOT_DIR \
  --sweep_configs $SWEEP_CONFIGS \
  dump_sweep_config_obj

# Create sweep config files
# This script dumps the config files into $SWEEP_ROOT_DIR/hyperparameters dir
python scripts/manual_sweep.py \
  --sweep_name $SWEEP_NAME \
  --sweep_root_dir $SWEEP_ROOT_DIR \
  --sweep_configs $SWEEP_CONFIGS \
  create_exp_configs --sweep-configs $SWEEP_CONFIGS

SWEEP_PROGRAM=$(python scripts/read_json_field.py "sweep_cfg.json" "program")
chmod +x $SWEEP_PROGRAM

# Iterate over $SWEEP_ROOT_DIR/hyperparameters and run the experiments
for CONFIG_FILE in $SWEEP_ROOT_DIR/hyperparameters/*.json; do
  RUN_NAME=$(basename $CONFIG_FILE .json)
  HP_EXP_DIR=$SWEEP_ROOT_DIR/exps/$RUN_NAME
  mkdir -p $HP_EXP_DIR

  is_complete=$(python scripts/manual_sweep.py \
    --sweep_name $SWEEP_NAME \
    --sweep_root_dir $SWEEP_ROOT_DIR \
    --sweep_configs $SWEEP_CONFIGS \
    is_hp_run_complete --run_name $RUN_NAME)

  if [ "$is_complete" == "True" ]; then
    echo "=========> [Manual Sweeper] Skipping $RUN_NAME as it is already complete"
    continue
  fi

  echo "=========> [Manual Sweeper] Running $RUN_NAME"

  RUN_ID=$(python scripts/manual_sweep.py \
    --sweep_name $SWEEP_NAME \
    --sweep_root_dir $SWEEP_ROOT_DIR \
    --sweep_configs $SWEEP_CONFIGS \
    generate_deterministic_run_id --run_name $RUN_NAME)

  APP_MANUAL_SWEEP_HYPERPARAMETER_FILE=$CONFIG_FILE APP_SWEEP_ROOT_DIR=$SWEEP_ROOT_DIR RUN_ID=$RUN_ID ./$SWEEP_PROGRAM
done

python scripts/manual_sweep.py \
  --sweep_name $SWEEP_NAME \
  --sweep_root_dir $SWEEP_ROOT_DIR \
  --sweep_configs $SWEEP_CONFIGS \
  fail_if_sweep_not_complete

python scripts/manual_sweep.py \
  --sweep_name $SWEEP_NAME \
  --sweep_root_dir $SWEEP_ROOT_DIR \
  --sweep_configs $SWEEP_CONFIGS \
  save_best_config
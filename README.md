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
### Overview
Example usage including **training**, **prediction**, **analysis**, **multiple seed**
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
### Setup for development
#### Local machine
Install dependencies:
```shell
conda create -n hf_proj python=3.9
conda activate hf_proj
conda install pytorch torchvision torchaudio -c pytorch
pip install -r src/requirements.txt
```
Clone the repo:
```shell
git clone https://github.com/kazemnejad/pt_hf_base.git
cd pt_hf_base
```
Download dataset files:
```shell
cd repor
export WANDB_ENTITY=kzmnjd WANDB_PROJECT=pt_hf_base && wandb artifact get kzmnjd/pt_hf_base/data-scan-length_25:latest \
    --root data/
```
Run the code using PyCharm's configuration or:
```shell
export APP_DS_SPLIT=length_25
export APP_SEED=1
python src/main.py --configs 'configs/t5a_debug.jsonnet,configs/data/scan.jsonnet' train
```
#### Cluster
To create and launch experiments on Mila, this repo comes with a ready-to-use pipeline. 
This is an overview of how the pipeline works:
1. First, we create an experiment archive file on our local machine,
containing the code and a reference to the dataset we need. 
Then, we upload it to Wandb Cloud to obtain a unique ID for it. These are done by `scripts/upload_experiment.py` 
2. We download the experiment archive file and its corresponding dataset from Wandb Cloud using the unique ID.
3. Finally, we submit the job into the cluster.
The last two steps are done by `scripts/launch_experiment.py`.

**Setup (Only once):**

1. Login into Mila cluster and run the following on the login node.
2. Install the dependencies for `launch_experiment.py`
```shell
module load miniconda/3
conda create -n launch_exp python=3.9
conda activate launch_exp
pip install wandb
wandb login
```
3. Clone the repo:
```shell
git clone https://github.com/kazemnejad/pt_hf_base.git
cd pt_hf_base
```
4. Download the container image inside Mila's scratch folder.
```shell
module load singularity
mkdir -p $HOME/scratch/containers
cd $HOME/scratch/containers
singularity pull --arch amd64 library://kzmnjd/deeplr/pt:v7
```
5. Optional: Create an alias for `launch_experiment.py`
```shell
vim ~/.bashrc
# Append this line to end:
alias launch_exp='module load miniconda/3 && conda activate launch_exp && python $HOME/<path/to/repo>/scripts/launch_experiment.py'
```

#### Launch the experiment (for each experiment)
```shell
export WANDB_ENTITY=kzmnjd WANDB_PROJECT=pt_hf_base && launch_exp \
  --slurm-args "--gres=gpu:a100:1 --reservation=DGXA100 --partition=main -t 20:00:00 -c 4 --mem=32G" \
  --image pt_v7.sif \
  <experiment_id>
```
If you want to launch the job in an interactive mode (`salloc`) 
pass the `--interactive` flag in the command line arguments.

#### Upload experiment (for each experiment)
1. Code the model, experiment, and stuff :)
2. Upload the experiment:
```shell
export DS=scan SPLIT=random && python scripts/upload_experiment.py \
  --configs "configs/config1.jsonnet,configs/config2.jsonnet" \
  --commands "train,predict,combine_pred,analyze_all" \
  --dataset "data-$DS-$SPLIT" \
  --env-vars "APP_DS_SPLIT=$SPLIT,KEY1=VAL1" \
  --seeds 5 \
  --name "_experiment_name_postfix" # (optional)
```
It will create and upload the experiment bundle for you on wandb
and returns its unique id.

## Code structure
### Overview
- `configs/`: Directory to store experiment configurations, all in .jsonnet format.
- `data/`: Datasets are store here in the following format (not present in the Git):
```
- data
    - <dataset1_name>
        - <split1_name>
            - train.jsonl
            - validation.jsonl
            - test.jsonl
```
- `experiments/`: Directory for experiment files e.g. checkpoints or logs (not present in the Git):
- `scripts/`: General purpose, single file scripts.
- `src/`: The model source code
  - `analyzers/`: Classes that mainly analyze a train model, such as its predictions.
  - `common/`: Common classes, functions, and utilities.
  - `data/`: Classes for reading the data, tokenizing it, and converting to input_ids.
  - `models/`: Model classes. e.g. T5, GPT-Neo
  - `modules/`: General-purpose PyTorch `nn.Modules`'s used in models. 
  - `runtime/`: Classes that contains routines such as training, inference, etc.
  - `tokenization_utils`: Different classes of tokenizers.
  - `trainer/`: Sub-classes of Huggingface trainers with added functionalities.

Usage of `src/main.py`:
```shell
python src/main.py \
	--debug_mode \ # optional
	--configs "path/to/config1.jsonnet,path/to/config2.jsonnet" \
	<command> --command_arg1 val1 --command_arg2 val2
```

### Adding a new model
1. Create its class under `src/models/`
```python
from models.base_model import Model, HfModelConfig

@Model.register("gpt_neo")
class CausalGPTNeo(GPTNeoForCausalLM, Model):
        def __init__(
        self,
        config: Optional[HfModelConfig] = None,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs,
    ):
        super().__init__(config)
        self.handle_tokenizer(tokenizer)

# Only required if it's an HF model
@HfModelConfig.register("gpt_neo_config", "from_di")
class DiGPTNeoConfig(GPTNeoConfig, HfModelConfig):
    pass
```
The line `@Model.register("gpt_neo")` will remember this class, 
and it can be referenced by `gpt_neo`.

`DiGPTNeoConfig` is just a wrapper class for model's configuration, 
and this allows us to reference it using `gpt_neo_config`

2. Import it in `src/models/__init__.py`
```shell
...

from .gpt_neo import CausalGPTNeo
```
3. Create a config for it and use it in experiments:
Example: `configs/models/gpt_neo_base.jsonnet`
```jsonnet
local base = (import 'base.jsonnet');

base + {
    model+: {
        type: 'gpt_neo',
        hf_model_name: 'EleutherAI/gpt-neo-125M',
        config+: {
            type: 'gpt_neo_config',
            hf_model_name: 'EleutherAI/gpt-neo-125M',
        },
        from_pretrained: true,
    },
}
```
Include in your experiment file:
Example: `configs/my_experiment_with_gpt_neo.jsonnet`
```jsonnet
...
+ (import 'configs/models/gpt_neo_base.jsonnet')
```
or pass it directly when running `main.py`
```shell
python src/main.py \
    --configs 'configs/<my_other_config1>,...,configs/models/gpt_neo_base.jsonnet' \
    train
```
### Dependency Injection
In this codebase, we use AllenNLP's dependency injection framework. Learn more at https://guide.allennlp.org/using-config-files

import argparse
import json
import os
import shlex
import site
import subprocess
import tempfile
from pathlib import Path

site.addsitedir("src/")

from wandb import env as wandb_env


os.environ[wandb_env.SILENT] = "true"
os.environ[wandb_env.DISABLE_CODE] = "true"


def make_run_script(
    configs: str, commands: str, env_vars: str, exp_key: str, exp_name: str
) -> Path:
    script = "#!/bin/bash\n\n\n"

    if env_vars:
        for ev in env_vars.split(","):
            ev = ev.strip()
            script += f"export {ev}\n"

    script += f"export WANDB_RUN_ID={exp_key}\n"
    script += f"export APP_EXPERIMENT_NAME={exp_name}\n"

    script = add_python_paths(script)

    configs_str = configs
    script += "\n\n"
    for c in commands.split(","):
        c = c.strip()
        script += f"python src/main.py --configs '{configs_str}' \\\n"
        script += f"       {c}\n\n"

    script += 'echo "Experiment finished!"\n'

    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / "run.sh"
    with open(script_path, "w") as f:
        f.write(script)

    subprocess.check_call(shlex.split(f"vim {script_path}"))

    return script_path


def add_python_paths(script):
    script += (
        "\n\nexport PYTHONPATH=$HOME/.local/lib/python3.6/site-packages/:$PYTHONPATH\n"
    )
    script += (
        "export PYTHONPATH=$HOME/.local/lib/python3.7/site-packages/:$PYTHONPATH\n"
    )
    script += (
        "export PYTHONPATH=$HOME/.local/lib/python3.8/site-packages/:$PYTHONPATH\n"
    )
    script += (
        "export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages/:$PYTHONPATH\n"
    )
    script += "\n\n#pip install --user -r src/requirements.txt\n"
    script += "\n\npip install --user scipy\n"
    return script


def make_run_script_seeds(
    configs: str,
    commands: str,
    env_vars: str,
    exp_key: str,
    exp_name: str,
    seeds: int,
) -> Path:
    script = "#!/bin/bash\n\n\n"

    if env_vars:
        for ev in env_vars.split(","):
            ev = ev.strip()
            script += f"export {ev}\n"

    # script += f"export WANDB_RUN_ID={exp_key}\n"
    script += f"export WANDB_RUN_GROUP=SE-{exp_name}\n"
    script += f"export WANDB_JOB_TYPE=training_seed\n"
    script += f"export ORIG_APP_EXPERIMENT_NAME={exp_name}\n"
    script += f"export ORIG_WANDB_RUN_ID={exp_key}\n"

    script = add_python_paths(script)
    script += "\n\n"

    configs_str = configs
    script += f"for SEED in `seq 1 {seeds}`; do\n"
    script += f"\texport APP_DIRECTORY=experiments/{exp_name}\n"
    script += f"\texport APP_EXPERIMENT_NAME=seed_$SEED\n"
    script += f"\texport APP_SEED=$SEED\n\n"
    for c in commands.split(","):
        c = c.strip()
        script += f"\tpython src/main.py --configs '{configs_str}' \\\n"
        script += f"\t       {c}\n\n"

    script += "done\n"

    script += '\necho "Experiment finished!"\n'

    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / "run.sh"
    with open(script_path, "w") as f:
        f.write(script)

    subprocess.check_call(shlex.split(f"vim {script_path}"))

    return script_path


def make_run_script_sweep_job(
    configs: str,
    commands: str,
    env_vars: str,
    exp_key: str,
    exp_name: str,
    seeds: int,
) -> Path:
    script = "#!/bin/bash\n\n\n"

    configs_str = configs
    for c in commands.split(","):
        c = c.strip()
        script += f"python src/main.py --configs '{configs_str}' \\\n"
        script += f"       {c}\n\n"

    script += '\necho "Experiment finished!"\n'

    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / "job.sh"
    with open(script_path, "w") as f:
        f.write(script)

    # subprocess.check_call(shlex.split(f"vim {script_path}"))

    return script_path


def make_run_script_sweep_agent(
    configs: str,
    commands: str,
    env_vars: str,
    exp_key: str,
    exp_name: str,
    sweep_id: str,
) -> Path:
    script = "#!/bin/bash\n\n\n"

    if env_vars:
        for ev in env_vars.split(","):
            ev = ev.strip()
            script += f"export {ev}\n"

    script = add_python_paths(script)
    script += "\n\n"

    sweep_key = os.path.basename(sweep_id)

    script += f"\nexport WANDB_RUN_GROUP=sweep-{sweep_key}\n"
    script += f"export WANDB_DIR=experiments/wandb_sweep_{sweep_key}\n"
    script += f"mkdir -p $WANDB_DIR\n"

    script += f"ln -srnf experiments/wandb_sweep_{sweep_key} experiments/{exp_name}/wandb_sweep_{sweep_key}\n"

    script += f"\nchmod a+x ./job.sh\n"
    script += f"wandb agent {sweep_id}\n"

    script += '\necho "Experiment finished!"\n'

    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / "run.sh"
    with open(script_path, "w") as f:
        f.write(script)

    return script_path


def make_metadata(exp_name, exp_key):
    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = tmp_dir / "metadata.json"

    metadata = {"exp_name": exp_name, "exp_key": exp_key}

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata_path


def unique_experiment_name_from_filenames(config_filenames):
    for p in config_filenames:
        assert os.path.exists(p)

    configs = "_".join(
        [os.path.splitext(os.path.basename(p))[0] for p in config_filenames]
    )

    unique_name = f"{configs}"

    return unique_name


def get_exp_name(configs):
    filenames = list(map(lambda x: x.strip(), configs.split(",")))
    exp_name = unique_experiment_name_from_filenames(filenames)
    return exp_name


def main(args: argparse.Namespace):
    project: str = args.project
    configs: str = args.configs

    print("# ----> 1. Generating a unique experiment name...")
    exp_name = get_exp_name(configs)

    if args.dataset is not None:
        if not args.dataset.startswith("data-"):
            args.dataset = f"data-{args.dataset}"

        ds_name = args.dataset.replace("/", "_")
        exp_name += f"___{ds_name}"

    if args.name is not None:
        exp_name += args.name

    group = None
    if args.seeds is not None:
        group = f"SE-{exp_name}"

    import wandb

    dir_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    dir_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=project,
        dir=dir_dir,
        group=group,
        name=exp_name,
        config={},
        mode="online",
        force=True,
        save_code=False,
        settings=wandb.Settings(disable_code=True, disable_git=True, silent=True),
    )
    run_id = run.id

    job_script_path = None
    if args.seeds is not None:
        run_script_path = make_run_script_seeds(
            configs, args.commands, args.env_vars, run_id, exp_name, args.seeds
        )
    elif args.sweep_id is not None:
        run_script_path = make_run_script_sweep_agent(
            configs, args.commands, args.env_vars, run_id, exp_name, args.sweep_id
        )
        job_script_path = make_run_script_sweep_job(
            configs, args.commands, args.env_vars, run_id, exp_name, args.sweep_id
        )
    else:
        run_script_path = make_run_script(
            configs, args.commands, args.env_vars, run_id, exp_name
        )

    metadata_path = make_metadata(exp_name, run_id)

    artifact_name = f"bundle-{run_id}"
    artifact = wandb.Artifact(name=artifact_name, type="bundle")
    artifact.add_dir("configs", "configs/")
    artifact.add_dir("src", "src/")
    artifact.add_dir("scripts", "scripts/")
    artifact.add_file(str(run_script_path), "run.sh")
    artifact.add_file(str(metadata_path), "metadata.json")
    if job_script_path is not None:
        artifact.add_file(str(job_script_path), "job.sh")

    artifact.metadata["data"] = args.dataset

    run.log_artifact(artifact)
    run.finish()

    print(f"\n\nExp name: {exp_name}")
    print(f"\n\nExp Key: {run_id}")
    print(f"Exp URL: {run.url}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make Experiment Bundle")

    parser.add_argument(
        "-s",
        "--configs",
        metavar="CONFIGS[,CONFIGS,CONFIGS]",
        type=str,
        help="Config file names",
    )

    parser.add_argument(
        "-c",
        "--commands",
        metavar="cmd -a -b[,cmd -c -d]",
        type=str,
        help="Experiment commands",
    )

    parser.add_argument(
        "-d", "--dataset", metavar="DATASET", type=str, help="Dataset name's bundle"
    )

    parser.add_argument(
        "-p",
        "--project",
        metavar="project",
        type=str,
        default="comp-gen_v2",
        help="Wandb project",
    )

    parser.add_argument(
        "-e",
        "--env-vars",
        metavar="KEY=VAL[,KEY=VAL]",
        type=str,
        help="Experiment environment variables",
    )

    parser.add_argument(
        "--seeds",
        metavar="NUM_SEEDS",
        type=int,
        help="Num of seeds",
    )

    parser.add_argument(
        "--sweep_id",
        metavar="SWEEP_ID",
        type=str,
        help="Wandb sweep id",
    )

    parser.add_argument(
        "-n",
        "--name",
        metavar="NAME",
        type=str,
        help="Name postfix",
    )

    args = parser.parse_args()

    main(args)

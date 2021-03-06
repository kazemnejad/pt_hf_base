import argparse
import os
import shlex
import site
import subprocess
import tempfile
from pathlib import Path

from wandb.sdk.lib import filenames

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
    script += f"export ORIG_APP_EXPERIMENT_NAME={exp_name}\n"
    script += f"export ORIG_WANDB_RUN_ID={exp_key}\n"

    script = add_python_paths(script)
    script += "\n\n"

    configs_str = configs
    script += f"for SEED in `seq 1 {seeds}`; do\n"
    script += f"\texport APP_DIRECTORY=experiments/{exp_name}\n"
    script += f"\texport APP_EXPERIMENT_NAME=seed_$SEED\n"
    script += f"\texport APP_SEED=$SEED\n"
    script += f"\texport WANDB_JOB_TYPE=exp\n"
    script += f"\texport WANDB_RUN_ID={exp_key}_seed_$SEED\n\n"

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
    script += "\nexport WANDB_JOB_TYPE=exp\n\n\n"

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
    script += f"export SWEEP_ID={sweep_id}\n"
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


def add_source_code(art):
    root = "src/"
    root = os.path.abspath(root)
    exclude_fn = lambda path: path.endswith(".pyc") or path.endswith("__pycache__")
    for file_path in filenames.filtered_dir(root, lambda p: True, exclude_fn):
        save_name = os.path.relpath(file_path, root)
        art.add_file(file_path, name=f"src/{save_name}")


def main(args: argparse.Namespace):
    project: str = args.project
    entity: str = args.entity
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

    import wandb

    group = "general"
    job_type = "exp"
    if args.seeds is not None:
        job_type = "seed_launcher"
        group = f"SE-{exp_name}"
    elif args.sweep_id is not None:
        job_type = "agent"
        group = f"sweep-{os.path.basename(args.sweep_id)}"

    dir_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    dir_dir.mkdir(parents=True, exist_ok=True)
    settings = wandb.Settings()
    settings.update(
        disable_code=True,
        disable_git=True,
        silent=True,
        _save_requirements=False,
        _disable_meta=True,
    )
    run = wandb.init(
        project=project,
        entity=entity,
        dir=dir_dir,
        group=group,
        name=exp_name,
        config={},
        mode="online",
        force=True,
        save_code=False,
        settings=settings,
        job_type=job_type,
        id=args.idx,
        resume="allow"
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
    artifact = wandb.Artifact(name=artifact_name, type="code")
    artifact.add_dir("configs", "configs/")
    add_source_code(artifact)
    artifact.add_dir("scripts", "scripts/")

    if os.path.exists(".run"):
        artifact.add_dir(".run", ".run/")
    if os.path.exists(".vscode"):
        artifact.add_dir(".vscode", ".vscode/")

    artifact.add_file(str(run_script_path), "run.sh")
    artifact.add_file(str(metadata_path), "metadata.json")
    if job_script_path is not None:
        artifact.add_file(str(job_script_path), "job.sh")

    if args.dataset is not None:
        artifact.metadata["data"] = args.dataset

    run.log_artifact(artifact)

    if args.dataset is not None:
        data_art_name = args.dataset
        if ":" not in data_art_name:
            data_art_name += ":latest"
        run.use_artifact(data_art_name)

    run.finish()

    print(f"\n\nExp name: {exp_name}")
    print(f"\n\nExp Key: {run_id}")
    print(f"Exp URL: {run.url}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make Experiment Bundle")

    if os.path.exists("configs/project_name.json"):
        with open("configs/project_name.json") as f:
            import json
            default_proj_name = json.load(f)["project_name"]
    else:
        default_proj_name = None

    if os.path.exists("configs/entity_name.json"):
        with open("configs/entity_name.json") as f:
            import json
            default_entity_name = json.load(f)["entity_name"]
    else:
        default_entity_name = None

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
        default=default_proj_name,
        help="Wandb project",
    )

    parser.add_argument(
        "--entity",
        metavar="entity",
        type=str,
        default=default_entity_name,
        help="Wandb entity",
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

    parser.add_argument(
        "-i",
        "--idx",
        metavar="IDX",
        type=str,
        help="Experiment Idx",
    )

    args = parser.parse_args()

    main(args)

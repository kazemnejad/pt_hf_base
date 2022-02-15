import os
from pathlib import Path
from typing import Dict, Any, List

from wandb.util import generate_id as wandb_generate_id

from common.nest import unflatten
from common.py_utils import unique_experiment_name
from runtime import Runtime

import logging
import sys

import fire

import json
import _jsonnet

from common import py_utils, Params, JsonDict

logger = logging.getLogger("app")
LOG_FORMAT = "%(levelname)s:%(name)-5s %(message)s"
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(py_utils.NewLineFormatter(LOG_FORMAT))
logger.addHandler(handler)

DEFAULT_SEED = "123"


class EntryPoint(object):
    _exp = None
    _config = None

    def __init__(self, configs: str, debug_mode: bool = None):
        filenames = [f.strip() for f in configs.split(",")]

        config = self._load_config_obj(filenames)

        if debug_mode is not None:
            if "global_vars" not in config:
                config["global_vars"] = dict()
            config["global_vars"]["debug_mode"] = debug_mode

        if config.get("sweep_run", False):
            config = self._patch_config_obj_for_sweep(config)
        else:
            config["exp_name"] = os.environ.get(
                "APP_EXPERIMENT_NAME", unique_experiment_name(config)
            )

        config_str = json.dumps(config, indent=4, sort_keys=True)
        logger.info(f"# configs: {filenames}")
        logger.info(f"----Config----\n{config_str}\n--------------")

        self._dump_config_obj(config)

        config = self._patch_config_obj_for_di(config)

        self._config = config
        self._exp = Runtime.from_params(Params({"config_dict": config, **config}))

    def _patch_config_obj_for_sweep(self, config: JsonDict) -> JsonDict:
        import wandb
        from wandb import env as wandb_env

        sweep_id = os.environ[wandb_env.SWEEP_ID]
        run_id = os.environ[wandb_env.RUN_ID]

        orig_exp_name = os.environ.get(
            "APP_EXPERIMENT_NAME", unique_experiment_name(config)
        )

        base_dir = Path(config.get("directory", "experiments"))
        sweep_root = base_dir / f"wandb_sweep_{sweep_id}"

        exps_dir = sweep_root / "exps"
        exps_dir.mkdir(parents=True, exist_ok=True)

        config["directory"] = str(exps_dir)
        config["exp_name"] = run_id

        run = wandb.init(name=f"{orig_exp_name}___{run_id}", allow_val_change=True)
        new_hyperparams = run.config.as_dict()
        new_hyperparams = {k:v for k,v in new_hyperparams.items() if not k.startswith("_wandb")}
        new_hyperparams = unflatten(new_hyperparams, ".")
        logger.info(f"New hyperparams: {new_hyperparams}")

        jsonnet_str = f"""
                    local base = {json.dumps(config)};
                    local diff = {new_hyperparams}; 
                    std.mergePatch(base, diff)
                    """
        patched_config = _jsonnet.evaluate_snippet("snippet", jsonnet_str)
        patched_config = json.loads(patched_config)

        run.config.update(patched_config)


        return patched_config

    def _patch_config_obj_for_di(self, config):
        if "runtime_type" in config:
            config["type"] = config["runtime_type"]
            del config["runtime_type"]
        return config

    def _dump_config_obj(self, config):
        exp_root = Path(config.get("directory", "experiments")) / config["exp_name"]
        exp_root.mkdir(parents=True, exist_ok=True)
        json.dump(config, (exp_root / "config.json").open("w"), indent=4, sort_keys=True)

    def _load_config_obj(self, filenames: List[str]) -> Dict[str, Any]:
        ext_vars = {k: v for k, v in os.environ.items() if k.startswith("APP_")}
        seed = os.environ.get("APP_SEED", DEFAULT_SEED)
        if not seed.isnumeric():
            seed = DEFAULT_SEED
        ext_vars["APP_SEED"] = seed
        jsonnet_str = "+".join([f'(import "{f}")' for f in filenames])
        json_str = _jsonnet.evaluate_snippet("snippet", jsonnet_str, ext_vars=ext_vars)
        config: Dict[str, Any] = json.loads(json_str)
        config["config_filenames"] = filenames

        orig_directory = config.get("directory", "experiments")
        config["directory"] = os.environ.get("APP_DIRECTORY", orig_directory)

        return config

    def __getattr__(self, attr):
        if attr in self.__class__.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self._exp, attr)

    def __dir__(self):
        return sorted(set(super().__dir__() + self._exp.__dir__()))


if __name__ == "__main__":
    fire.Fire(EntryPoint)

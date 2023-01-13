import json
from pathlib import Path

from wandb.sdk.wandb_run import Run

from common import Registrable
from data import DataLoaderFactory
from models import Model


class Analyzer(Registrable):
    def __init__(
        self,
        model: Model,
        logger: Run,
        dl_factory: DataLoaderFactory,
        exp_root: Path,
        split: str,
        **kwargs
    ):
        self.model = model
        self.logger = logger
        self.dl_factory = dl_factory
        self.exp_root = exp_root
        self.split = split

        self._local_log_obj = {}

    def analyze(self):
        pass

    def log(self, obj):
        self._local_log_obj.update(obj)

    def flush_local_log(self):
        analysis_name = self.__class__.__name__ + "__" + self.split
        analysis_root = self.exp_root / "analysis" / analysis_name
        analysis_root.mkdir(parents=True, exist_ok=True)

        with (analysis_root / "log.json").open("w") as f:
            json.dump(self._local_log_obj, f, indent=4)

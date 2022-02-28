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
        **kwargs
    ):
        self.model = model
        self.logger = logger
        self.dl_factory = dl_factory
        self.exp_root = exp_root

    def analyze(self):
        pass

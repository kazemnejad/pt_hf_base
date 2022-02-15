import abc
import logging
from pathlib import Path
from typing import Union, Any, Callable, Optional, Dict, List

from torch.utils.data import DataLoader, Dataset, IterableDataset

from common import Registrable, FromParams, Lazy, ConfigurationError, ExperimentStage

logger = logging.getLogger("app")

DatasetType = Union[Dataset, IterableDataset]


class DIDataloader(DataLoader, FromParams):
    pass


class DataLoaderFactory(Registrable):
    def __init__(
        self,
        data_root: str,
        name: Optional[str] = None,
        split: Optional[str] = None,
        directory: Optional[str] = None,
        train_filename: Optional[str] = "train.tsv",
        validation_filename: Optional[str] = "valid.tsv",
        test_filename: Optional[str] = "test.tsv",
        train_batch_size: int = 2,
        validation_batch_size: int = 2,
        test_batch_size: int = 2,
        shuffle: bool = False,
        dataloader: Optional[Lazy[DIDataloader]] = None,
    ):
        data_root = Path(data_root)

        if directory:
            self.dataset_dir = data_root / directory
        else:
            if name is None:
                raise ConfigurationError(
                    f"either (name,split) or directory has to be defined"
                )

            self.dataset_dir = data_root / name
            if split is not None:
                self.dataset_dir /= split

        self.train_filename = train_filename
        self.validation_filename = validation_filename
        self.test_filename = test_filename

        self.train_batch_size = train_batch_size
        self.valid_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size

        self.shuffle = shuffle
        self.dataloader = dataloader or Lazy(DIDataloader)

        self._ds_cache: Dict[str, Any] = {}

        self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def get_ds_file_path(self, stage: ExperimentStage = None, path: str = None) -> str:
        if path is None:
            file_path = self.dataset_dir
            if stage == ExperimentStage.TRAINING:
                file_path /= self.train_filename
            elif stage == ExperimentStage.VALIDATION:
                file_path /= self.validation_filename
            elif stage == ExperimentStage.TEST:
                file_path /= self.test_filename

            if not file_path.exists():
                raise ValueError(f"Path {file_path} does not exist")

            return str(file_path)
        else:
            return path

    def get_train_dataset(self, force_rebuild: bool = False) -> DatasetType:
        return self.get_dataset(
            stage=ExperimentStage.TRAINING, force_rebuild=force_rebuild
        )

    def get_validation_dataset(self, force_rebuild: bool = False) -> DatasetType:
        return self.get_dataset(
            stage=ExperimentStage.VALIDATION, force_rebuild=force_rebuild
        )

    def get_test_dataset(self, force_rebuild: bool = False) -> DatasetType:
        return self.get_dataset(stage=ExperimentStage.TEST, force_rebuild=force_rebuild)

    def _get_batch_size(self, stage):
        if stage == ExperimentStage.TRAINING:
            return self.train_batch_size
        elif stage == ExperimentStage.VALIDATION:
            return self.valid_batch_size
        elif stage == ExperimentStage.TEST:
            return self.test_batch_size

    @abc.abstractmethod
    def get_collate_fn(self, state: ExperimentStage) -> Callable:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_compute_metric_fn(self, stage: ExperimentStage) -> Callable:
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_line_to_instance(self, line: str, stage: ExperimentStage) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_model_output_to_line(self, outputs: Any) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def build_dataset(
        self, path: Path, stage: ExperimentStage, **kwargs
    ) -> DatasetType:
        raise NotImplementedError()

    def get_dataset(
        self,
        stage: Optional[ExperimentStage] = None,
        path: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> DatasetType:
        assert stage is not None or path is None

        if path is None:
            ds_path = self.dataset_dir
            if stage == ExperimentStage.TRAINING:
                ds_path /= self.train_filename
            elif stage == ExperimentStage.VALIDATION:
                ds_path /= self.validation_filename
            elif stage == ExperimentStage.TEST:
                ds_path /= self.test_filename
            else:
                raise ValueError(f"Unsupported stage = {stage}")
        else:
            ds_path = Path(path)

        if str(ds_path) in self._ds_cache and not force_rebuild:
            return self._ds_cache[str(ds_path)]

        if not ds_path.exists():
            logger.warning(f"The dataset at {ds_path} doesn't exist. Skipping ...")
            return None

        ds = self.build_dataset(ds_path, stage)
        self._ds_cache[str(ds_path)] = ds

        return ds

    def get_column_names(self) -> List[str]:
        raise NotImplementedError()

    def build(
        self,
        stage: ExperimentStage = ExperimentStage.TRAINING,
        path: Optional[str] = None,
    ) -> DataLoader:
        dataset = self.get_dataset(stage=stage, path=path)

        shuffle = self.shuffle
        shuffle &= stage == ExperimentStage.TRAINING

        dataloder = self.dataloader.construct(
            dataset=dataset,
            batch_size=self._get_batch_size(stage),
            collate_fn=self.get_collate_fn(stage),
            pin_memory=True,
            drop_last=False,
            shuffle=shuffle,
        )

        return dataloder


DataLoaderFactory.register("base")(DataLoaderFactory)

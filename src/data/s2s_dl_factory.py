import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple, Deque

import datasets as hf_datasets
import numpy as np
from datasets import set_progress_bar_enabled, Dataset
from overrides import overrides
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    EvalPrediction,
    AddedToken,
)

from common import ExperimentStage
from common import Lazy, Params
from common.py_utils import chunks
from data.base_dl_factory import DataLoaderFactory
from tokenization_utils import SpecialTokens, Tokenizer

logger = logging.getLogger("app")

DEFAULT_EOS_TOKEN = "</s>"


@DataLoaderFactory.register("seq2seq")
class Seq2SeqDataLoaderFactory(DataLoaderFactory):
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        source_seq_key: Optional[str] = "source",
        target_seq_key: Optional[str] = "target",
        src_vocab_file: Optional[str] = "vocab.src.txt",
        tgt_vocab_file: Optional[str] = "vocab.tgt.txt",
        append_vocab: Optional[str] = "no",
        max_source_length: Optional[int] = 100,
        max_target_length: Optional[int] = 100,
        hf_ds: Optional[Lazy[hf_datasets.Dataset]] = None,
        num_proc: Optional[int] = os.cpu_count() // 2,
        enable_hf_datasets_cache: Optional[bool] = False,
        **kwargs,
    ):
        hf_datasets.set_caching_enabled(enable_hf_datasets_cache)
        super().__init__(**kwargs)

        self.cache_dir = cache_dir

        self.source_seq_key = source_seq_key
        self.target_seq_key = target_seq_key

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.append_vocab = append_vocab
        self.src_vocab_file = src_vocab_file
        self.tgt_vocab_file = tgt_vocab_file

        self.hf_ds = hf_ds or Lazy(
            hf_datasets.load_dataset,
            constructor_extras={
                "path": "csv",
                "delimiter": "\t",
                "column_names": ("source", "target"),
                "download_mode": "force_redownload",
            },
        )

        self.num_proc = num_proc

    def set_tokenizer(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

        add_vocab = self.append_vocab in ["all", "src", "tgt"]
        if not add_vocab:
            return

        def read_vocab(vocab_path: Path) -> List[AddedToken]:
            v = []
            with open(vocab_path) as f:
                for w in f:
                    w = w.strip()
                    if w.lower() in SpecialTokens.all() + ["<eos>", "[pad]"]:
                        continue
                    v.append(AddedToken(w, single_word=True))
            return v

        if self.append_vocab in ["src", "all"]:
            logger.info("Appending source vocab to the tokenizer...")
            self.tokenizer.add_tokens(
                read_vocab(self.dataset_dir / self.src_vocab_file)
            )

        if self.append_vocab in ["tgt", "all"]:
            logger.info("Appending target vocab to the tokenizer...")
            self.tokenizer.add_tokens(
                read_vocab(self.dataset_dir / self.tgt_vocab_file)
            )

    def _get_tokenize_function(self, add_special_tokens: bool = True) -> Callable:
        tokenizer = self.tokenizer
        max_source_length = self.max_source_length
        max_target_length = self.max_target_length
        src_seq_key = self.source_seq_key
        tgt_seq_key = self.target_seq_key

        def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
            inputs = example[src_seq_key]
            targets = example[tgt_seq_key]

            encoding = tokenizer(
                inputs,
                padding="longest",
                max_length=max_source_length,
                truncation=True,
                add_special_tokens=add_special_tokens,
                return_tensors="pt",
            )
            input_ids, attention_mask = (
                encoding.input_ids[0],
                encoding.attention_mask[0],
            )

            # encode the targets
            target_encoding = tokenizer(
                targets,
                padding="longest",
                max_length=max_target_length,
                add_special_tokens=add_special_tokens,
                truncation=True,
            )
            labels = target_encoding.input_ids
            labels = [
                label if label != tokenizer.pad_token_id else -100 for label in labels
            ]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        return tokenize

    def tokenize_function(self, example: Dict[str, Any]) -> Dict[str, Any]:
        tokenizer = self.tokenizer
        max_source_length = self.max_source_length
        max_target_length = self.max_target_length
        src_seq_key = self.source_seq_key
        tgt_seq_key = self.target_seq_key

        inputs = example[src_seq_key]
        targets = example[tgt_seq_key]

        encoding = tokenizer(
            inputs,
            padding="longest",
            max_length=max_source_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = (
            encoding.input_ids[0],
            encoding.attention_mask[0],
        )

        # encode the targets
        target_encoding = tokenizer(
            targets,
            padding="longest",
            max_length=max_target_length,
            add_special_tokens=True,
            truncation=True,
        )
        labels = target_encoding.input_ids
        labels = [
            label if label != tokenizer.pad_token_id else -100 for label in labels
        ]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @overrides
    def transform_line_to_instance(self, line: str, stage: ExperimentStage) -> Any:
        parts = line.strip().split("\t")
        source = parts[0]
        if len(parts) == 2:
            target = parts[1]
        else:
            target = []

        instance = {
            "source": source,
            "target": target,
        }
        instance.update(self._get_tokenize_function()(instance))

        return instance

    def build_dataset(
        self,
        path: Path,
        stage: ExperimentStage,
        add_idx: bool = True,
        tokenize: bool = True,
        **kwargs,
    ) -> Dataset:
        ds = self._build_base_dataset(path)

        set_progress_bar_enabled(False)

        # Add index
        if add_idx and "idx" not in ds.features:
            ds = ds.map(
                lambda example, idx: {"idx": idx},
                with_indices=True,
                load_from_cache_file=False,
                keep_in_memory=True,
            )

        # Tokenize sequences and map tokens to their token ids
        if tokenize and any(f not in ds.features for f in ("input_ids", "labels")):
            ds = ds.map(
                self._get_tokenize_function(),
                num_proc=os.cpu_count(),
                load_from_cache_file=False,
                keep_in_memory=True,
            )

        set_progress_bar_enabled(True)

        return ds

    def _build_base_dataset(self, path):
        default_hf_ds_kwargs = {
            "data_files": {path.name: [str(path)]},
            "split": path.name,
            "cache_dir": self.cache_dir,
        }
        ds = self.hf_ds.construct(
            **{
                k: v
                for k, v in default_hf_ds_kwargs.items()
                if k not in self.hf_ds._constructor_extras
            }
        )
        return ds

    @overrides
    def get_column_names(self) -> List[str]:
        return ["input_ids", "attention_mask", "labels"]

    @overrides
    def get_collate_fn(self, state: ExperimentStage) -> Callable:
        collator = DataCollatorForSeq2Seq(
            self.tokenizer, label_pad_token_id=-100, padding="longest"
        )
        return collator

    def get_compute_metric_fn_for_train(
        self,
    ) -> Callable:
        tokenizer = self.tokenizer

        metric_funcs = {
            "f1": padded_f1,
            "recall": padded_recall,
            "precision": padded_precision,
            "acc": padded_accuracy,
            "seq_acc": padded_sequence_accuracy,
        }

        def compute_metrics(eval_preds: EvalPrediction):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]

            if len(preds.shape) == 3:
                preds = np.argmax(preds, axis=-1)

            if np.all(preds[:, 0] == 0):
                preds = preds[:, 1:]

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            num_all_examples = preds.shape[0]
            batch_size = 512

            all_metrics: Dict[str, Deque[float]] = defaultdict(deque)
            for batch_preds, batch_labels in tqdm(
                zip(chunks(preds, batch_size), chunks(labels, batch_size)),
                total=len(preds) // batch_size,
            ):
                weight = batch_preds.shape[0]
                padded_preds, padded_labels = pad_tensors_to_same_length(
                    batch_preds, batch_labels
                )
                for metric_name, metric_fn in metric_funcs.items():
                    result = metric_fn(
                        padded_preds=padded_preds, padded_labels=padded_labels
                    )
                    all_metrics[metric_name].append(result * weight)

            final_result = {
                k: sum(v) / num_all_examples for k, v in all_metrics.items()
            }

            final_result = {k: round(v, 4) for k, v in final_result.items()}

            return final_result

        return compute_metrics

    def get_compute_metric_fn_for_predict(self) -> Callable:
        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            return preds, labels

        def sequence_accuracy(preds: List[str], labels: List[str]):
            assert len(preds) == len(labels)

            correct_seqs = [p for (p, l) in zip(preds, labels) if p == l]
            return len(correct_seqs) / len(labels)

        def accuracy(preds: List[str], labels: List[str]):
            num_correct = 0
            num_totals = 0
            for p, l in zip(preds, labels):
                p = p.split()
                l = l.split()

                if len(p) > len(l):
                    p = p[: len(l)]
                else:
                    p = p + ["<PAD_TOKEN>"] * (len(l) - len(p))

                num_correct += len([1 for t1, t2 in zip(p, l) if t1 == t2])
                num_totals += len(l)

            return num_correct / num_totals

        tokenizer = self.tokenizer

        def compute_metrics_for_batch(preds, labels) -> tuple[Any, Any]:
            decoded_preds = tokenizer.batch_decode(
                preds, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            decoded_labels = tokenizer.batch_decode(
                labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            result = {
                "seq_acc": sequence_accuracy(decoded_preds, decoded_labels) * 100,
                # "acc": accuracy(decoded_preds, decoded_labels) * 100,
            }

            result = {k: round(v, 4) for k, v in result.items()}

            return result["seq_acc"]

        def compute_metrics(eval_preds: EvalPrediction):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]

            if len(preds.shape) == 3:
                preds = np.argmax(preds, axis=-1)

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            seq_accs = []
            accs = []
            batch_size = 10
            for batch_preds, batch_labels in tqdm(
                zip(chunks(preds, batch_size), chunks(labels, batch_size)),
                total=len(preds) // batch_size,
            ):
                count = len(batch_preds)
                seq_acc = compute_metrics_for_batch(batch_preds, batch_labels)
                acc, weights = padded_accuracy(preds=batch_preds, labels=batch_labels)
                acc = acc.sum() / weights.sum() * 100
                seq_accs.append(seq_acc * count)
                accs.append(acc * count)

            result = {
                "seq_acc": sum(seq_accs) / len(preds),
                "acc": sum(accs) / len(preds),
            }

            result = {k: round(v, 4) for k, v in result.items()}

            return result

        return compute_metrics

    @overrides
    def get_compute_metric_fn(
        self, stage: ExperimentStage = ExperimentStage.PREDICTION
    ) -> Callable:
        if stage == ExperimentStage.TRAINING:
            return self.get_compute_metric_fn_for_train()
        else:
            return self.get_compute_metric_fn_for_train()


def pad_tensors_to_same_length(
    x: np.ndarray, y: np.ndarray, constant_values: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad x and y so that the results have the same length (second dimension)."""
    x_length = x.shape[1]
    y_length = y.shape[1]
    max_length = max(x_length, y_length)

    x_non_touching_dims = [[0, 0] for _ in range(len(x.shape) - 2)]
    y_non_touching_dims = [[0, 0] for _ in range(len(y.shape) - 2)]

    padding_dim = [[0, 0], [0, max_length - x_length], *x_non_touching_dims]
    x = np.pad(x, padding_dim, constant_values=constant_values)

    padding_dim = [[0, 0], [0, max_length - y_length], *y_non_touching_dims]
    y = np.pad(y, padding_dim, constant_values=constant_values)

    return x, y


def padded_sequence_accuracy(
    *, padded_preds: np.ndarray = None, padded_labels: np.ndarray = None
) -> float:
    weights = np.not_equal(padded_labels, 0).astype(np.float32)

    padded_preds = padded_preds.astype(np.int64)
    padded_labels = padded_labels.astype(np.int64)
    not_correct = np.not_equal(padded_preds, padded_labels).astype(np.float32) * weights
    axis = tuple(range(1, len(padded_preds.shape)))
    correct_seq: np.ndarray = 1.0 - np.minimum(1.0, np.sum(not_correct, axis=axis))

    seq_acc = (
        correct_seq.sum()
        / np.ones(shape=correct_seq.shape, dtype=correct_seq.dtype).sum()
    )

    return float(seq_acc)


def padded_recall(
    *, padded_preds: np.ndarray = None, padded_labels: np.ndarray = None
) -> float:
    # padded_preds, padded_labels = pad_tensors_to_same_length(padded_preds, padded_labels)
    weights = np.not_equal(padded_labels, 0).astype(np.float32)
    padded_preds = padded_preds.astype(np.int64)
    padded_labels = padded_labels.astype(np.int64)

    recall = np.equal(padded_preds, padded_labels).astype(np.float32) * weights
    recall = recall.sum() / weights.sum()

    return float(recall)


def padded_accuracy(
    *, padded_preds: np.ndarray = None, padded_labels: np.ndarray = None
) -> float:
    weights = np.ones_like(padded_labels).astype(np.float32)
    padded_preds = padded_preds.astype(np.int64)
    padded_labels = padded_labels.astype(np.int64)
    acc = np.equal(padded_preds, padded_labels).astype(np.float32).sum() / weights.sum()
    return float(acc)


def padded_precision(
    *, padded_preds: np.ndarray = None, padded_labels: np.ndarray = None
) -> float:
    weights = np.not_equal(padded_preds, 0).astype(np.float32)
    padded_preds = padded_preds.astype(np.int64)
    padded_labels = padded_labels.astype(np.int64)
    precision = np.equal(padded_preds, padded_labels).astype(np.float32) * weights
    precision = precision.sum() / weights.sum()
    return float(precision)


def padded_f1(
    *, padded_preds: np.ndarray = None, padded_labels: np.ndarray = None
) -> float:
    recall = padded_recall(padded_preds=padded_preds, padded_labels=padded_labels)
    precision = padded_precision(padded_preds=padded_preds, padded_labels=padded_labels)
    nom = precision * recall
    denom = precision + recall

    f1 = float(2.0 * (nom) / (denom)) if denom != 0 else 0

    return f1


if __name__ == "__main__":
    dl_factory = Seq2SeqDataLoaderFactory.from_params(
        Params(
            {
                "data_root": "data",
                "name": "scan",
                "split": "switch",
            }
        )
    )
    tokenizer = Lazy(
        Tokenizer,
        params=Params(
            {
                # "type": "pretrained",
                # "hf_model_name": "t5-small",
                # "use_fast": False,
                "type": "whitespace",
            }
        ),
    ).construct(dataset=dl_factory, experiment_root="experiments/base")

    dl_factory.set_tokenizer(tokenizer)

    stage = ExperimentStage.TRAINING
    ds = dl_factory.get_dataset(stage)
    ds = ds.with_format(type="pt", columns=["input_ids", "attention_mask", "labels"])
    print(ds)

    dc = DataCollatorForSeq2Seq(
        dl_factory.tokenizer, label_pad_token_id=-100, padding="longest"
    )

    b = [ds[i] for i in range(2)]

    print(dc(b, return_tensors="pt"))
    print(ds[0])

    # dataloader = dl_factory.build(stage)
    # dataloader = iter(dataloader)
    #
    # batch = next(dataloader)
    #
    # print(batch)

import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Dict, Any, Callable, Tuple, Deque, Optional

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
from common.py_utils import chunks
from data.base_dl_factory import DataLoaderFactory
from data.s2s_dl_factory import Seq2SeqDataLoaderFactory

logger = logging.getLogger("app")

from tokenization_utils import Tokenizer


SCRATCHPAD_BOS = "<scratch>"
SCRATCHPAD_EOS = "</scratch>"
SCRATCHPAD_SEP = "\n"


@DataLoaderFactory.register("addition_task", exist_ok=True)
class AdditionTaskDataLoaderFactory(Seq2SeqDataLoaderFactory):
    def __init__(
        self,
        include_scratchpad: bool,
        is_seq2seq: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
            train_filename="train.jsonl",
            validation_filename="valid.jsonl",
            test_filename="test.jsonl",
        )

        self.include_scratchpad = include_scratchpad
        self.is_seq2seq = is_seq2seq

    def set_tokenizer(self, tokenizer: Tokenizer):
        super(AdditionTaskDataLoaderFactory, self).set_tokenizer(tokenizer)

        self.tokenizer.add_tokens(AddedToken(SCRATCHPAD_BOS, single_word=True))
        self.tokenizer.add_tokens(AddedToken(SCRATCHPAD_EOS, single_word=True))

        if self.tokenizer._pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(0)

    def _get_tokenize_function(
        self, add_special_tokens: bool = True, is_training: bool = True
    ) -> Callable:
        tokenizer = self.tokenizer
        max_source_length = self.max_source_length
        max_target_length = self.max_target_length
        src_seq_key = self.source_seq_key
        tgt_seq_key = self.target_seq_key

        is_seq2seq = self.is_seq2seq
        include_scratchpad = self.include_scratchpad

        def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
            inputs = example[src_seq_key].strip()
            targets = example[tgt_seq_key].strip()

            if include_scratchpad:
                scratchpad = SCRATCHPAD_SEP.join(example["scratchpad"])
                targets = f"{scratchpad}{SCRATCHPAD_SEP}{targets}"

            if is_seq2seq:
                encoding = tokenizer(
                    inputs,
                    max_length=max_source_length,
                    truncation=True,
                    add_special_tokens=add_special_tokens,
                )
                input_ids, attention_mask = (
                    encoding.input_ids[0],
                    encoding.attention_mask[0],
                )

                # encode the targets
                target_encoding = tokenizer(
                    targets,
                    truncation=True,
                    max_length=max_target_length,
                    add_special_tokens=add_special_tokens,
                )
                labels = target_encoding.input_ids
            else:
                inputs = f"{inputs}{SCRATCHPAD_SEP}"
                targets = f"{targets}{tokenizer.eos_token}"

                encoding = tokenizer(
                    inputs,
                    padding="longest",
                    max_length=max_source_length,
                    truncation=True,
                    add_special_tokens=add_special_tokens,
                )
                input_ids = encoding.input_ids

                encoding = tokenizer(
                    targets,
                    padding="longest",
                    max_length=max_target_length,
                    truncation=True,
                    add_special_tokens=add_special_tokens,
                )
                target_ids = encoding.input_ids

                labels = input_ids + target_ids

                if is_training:
                    input_ids = labels

                attention_mask = [1] * len(input_ids)

            labels = [
                label if label != tokenizer.pad_token_id else -100 for label in labels
            ]

            assert len(input_ids) == len(attention_mask)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        return tokenize

    def build_dataset(
        self,
        path: Path,
        stage: ExperimentStage,
        **kwargs,
    ) -> Dataset:
        ds = self._build_base_dataset(path)

        set_progress_bar_enabled(False)

        # Add index
        if "idx" not in ds.features:
            ds = ds.map(
                lambda example, idx: {"idx": idx},
                with_indices=True,
                load_from_cache_file=False,
                keep_in_memory=True,
            )

        # Tokenize sequences and map tokens to their token ids
        if any(f not in ds.features for f in ("input_ids", "labels")):
            ds = ds.map(
                self._get_tokenize_function(
                    is_training=stage == ExperimentStage.TRAINING
                ),
                num_proc=min(4, os.cpu_count()),
                load_from_cache_file=False,
                keep_in_memory=True,
            )

        set_progress_bar_enabled(True)

        return ds

    def _build_base_dataset(self, path):
        ds = Dataset.from_json(str(path))
        return ds

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
    from common import Params, Lazy

    dl_factory = AdditionTaskDataLoaderFactory.from_params(
        Params(
            {
                "data_root": "data",
                "name": "addition",
                "split": "normal",
                "include_scratchpad": False,
                "is_seq2seq": False,
                "max_source_length": 100,
                "max_target_length": 512,
            }
        )
    )
    tokenizer = Lazy(
        Tokenizer,
        params=Params(
            {
                "type": "pretrained",
                "hf_model_name": "EleutherAI/gpt-neo-125M",
                "use_fast": True,
                # "type": "whitespace",
            }
        ),
    ).construct(dataset=dl_factory, experiment_root="experiments/base")

    dl_factory.set_tokenizer(tokenizer)

    stage = ExperimentStage.TEST
    ds = dl_factory.get_dataset(stage)
    ds = ds.with_format(type="pt", columns=["input_ids", "attention_mask", "labels"])
    print(ds)

    ds = ds.remove_columns(["source", "target", "scratchpad", "_gen_src", "idx"])
    dc = DataCollatorForSeq2Seq(
        dl_factory.tokenizer, label_pad_token_id=-100, padding="longest"
    )

    b = [ds[i] for i in range(2)]

    print(dc(b, return_tensors="pt"))
    print(ds[0])

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset=ds,
        batch_size=10,
        collate_fn=dl_factory.get_collate_fn(stage),
        drop_last=False,
        shuffle=False,
    )

    dataloader = iter(dataloader)
    batch = next(dataloader)

    print(batch)

    # g = AdditionTaskGenerator(8, False, 100000, include_scratchpad=True)
    # o = g.generate()
    #
    # print("o")

import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple, Deque

import datasets as hf_datasets
import numpy as np
import torch
import wandb
from datasets import set_progress_bar_enabled, Dataset
from overrides import overrides
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    EvalPrediction,
    AddedToken,
)

from common import ExperimentStage, JsonDict
from common import Lazy, Params
from common.py_utils import chunks
from data.base_dl_factory import DataLoaderFactory
from tokenization_utils import SpecialTokens, Tokenizer

logger = logging.getLogger("app")

DEFAULT_EOS_TOKEN = "</s>"


@DataLoaderFactory.register("seq2seq", exist_ok=True)
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
        is_decoder_only: Optional[bool] = False,
        decoder_only_input_output_sep_token: Optional[str] = "<sep>",
        decoder_only_block_size: Optional[int] = 1024,
        decoder_only_group_samples: Optional[bool] = False,
        decoder_only_mask_inputs: Optional[bool] = True,
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

        self.is_decoder_only = is_decoder_only
        self.decoder_only_input_output_sep_token = decoder_only_input_output_sep_token
        self.decoder_only_block_size = decoder_only_block_size
        self.decoder_only_group_samples = decoder_only_group_samples
        self.decoder_only_mask_inputs = decoder_only_mask_inputs

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

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(0)

        if self.is_decoder_only:
            logger.info("Appending input_output separator token...")
            self.tokenizer.add_tokens(
                AddedToken(self.decoder_only_input_output_sep_token, single_word=False)
            )
            self.tokenizer.padding_side = "left"

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

        ds = self._build_tokenized_dataset(stage, ds, tokenize=tokenize)

        set_progress_bar_enabled(True)

        self._log_examples(ds, str(stage))

        return ds

    def _build_base_dataset(self, path: Path) -> Dataset:
        if str(path).endswith(".jsonl"):
            ds = Dataset.from_json(str(path))
        else:
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

    def _build_tokenized_dataset(
        self,
        stage: ExperimentStage,
        base_ds: Dataset,
        tokenize: bool = True,
    ) -> Dataset:
        ds = base_ds

        # Tokenize sequences and map tokens to their token ids
        if tokenize and any(f not in ds.features for f in ("input_ids", "labels")):
            ds = ds.map(
                self._get_tokenize_function(
                    add_special_tokens=True,
                    is_training=stage == ExperimentStage.TRAINING,
                ),
                num_proc=min(os.cpu_count(), 4),
                load_from_cache_file=False,
                keep_in_memory=True,
            )

        if (
            self.is_decoder_only
            and stage == ExperimentStage.TRAINING
            and self.decoder_only_group_samples
        ):
            ds = ds.remove_columns(
                [
                    c
                    for c in ds.column_names
                    if c not in ["input_ids", "labels", "attention_mask"]
                ]
            )
            ds = ds.map(
                self._get_group_fn(),
                batched=True,
                num_proc=min(4, os.cpu_count()),
                load_from_cache_file=False,
                desc=f"Grouping texts in chunks of {self.decoder_only_block_size}",
            )

        return ds

    def _get_tokenize_function(
        self, add_special_tokens: bool = True, is_training: bool = True
    ) -> Callable[[JsonDict], JsonDict]:
        if self.is_decoder_only:
            return self._get_tokenize_function_for_decoder_only(
                add_special_tokens=add_special_tokens, is_training=is_training
            )
        else:
            return self._get_tokenize_function_for_encoder_decoder(
                add_special_tokens=add_special_tokens
            )

    def _get_tokenize_function_for_encoder_decoder(
        self, add_special_tokens: bool = True
    ) -> Callable:
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

    def _get_tokenize_function_for_decoder_only(
        self, add_special_tokens: bool = True, is_training: bool = True
    ) -> Callable:
        tokenizer = self.tokenizer
        max_source_length = self.max_source_length
        max_target_length = self.max_target_length
        src_seq_key = self.source_seq_key
        tgt_seq_key = self.target_seq_key
        mask_inputs = self.decoder_only_mask_inputs

        def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
            inputs = example[src_seq_key].strip()
            targets = example[tgt_seq_key].strip()

            prompt = f"{inputs}{self.decoder_only_input_output_sep_token}"
            targets = f"{targets}{tokenizer.eos_token if add_special_tokens else ''}"
            sample = f"{prompt}{targets}"

            prompt_ids = tokenizer(
                prompt,
                truncation=True,
                add_special_tokens=False,
                max_length=max_source_length,
            ).input_ids

            sample_ids = tokenizer(
                sample,
                truncation=True,
                add_special_tokens=False,
                max_length=max_target_length + max_source_length,
            ).input_ids

            labels = sample_ids

            if is_training:
                input_ids = labels.copy()
            else:
                input_ids = prompt_ids

            if mask_inputs:
                prompt_ids_len = len(prompt_ids)
                labels = [-100] * prompt_ids_len + labels[prompt_ids_len:]

            attention_mask = [1] * len(input_ids)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        return tokenize

    def _get_group_fn(self):
        block_size = self.decoder_only_block_size

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        return group_texts

    def _log_examples(self, ds: Dataset, name: str):
        table = wandb.Table(
            columns=[
                "idx",
                "input_ids",
                "labels",
                "input_id_tokens",
                "label_tokens",
                "src",
                "tgt",
            ]
        )
        examples = ds[:100]

        for idx in range(len(examples[list(examples.keys())[0]])):

            input_ids = [tid for tid in examples["input_ids"][idx] if tid >= 0]
            labels = [tid for tid in examples["labels"][idx] if tid >= 0]
            if self.source_seq_key in examples:
                src = examples[self.source_seq_key][idx]
            else:
                src = ""

            if self.target_seq_key in examples:
                tgt = examples[self.target_seq_key][idx]
            else:
                tgt = ""

            decoded_input_ids = self.tokenizer.decode(input_ids)
            input_id_tokens = [
                self.tokenizer.convert_ids_to_tokens(tid) if tid >= 0 else str(tid)
                for tid in examples["input_ids"][idx]
            ]
            decoded_labels = self.tokenizer.decode(labels)
            label_tokens = [
                self.tokenizer.convert_ids_to_tokens(tid) if tid >= 0 else str(tid)
                for tid in examples["labels"][idx]
            ]

            table.add_data(
                idx,
                decoded_input_ids,
                decoded_labels,
                ", ".join(input_id_tokens),
                ", ".join(label_tokens),
                src,
                tgt,
            )

        if wandb.run is not None:
            wandb.log({f"ds_examples/{name}": table})

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

    @overrides
    def get_column_names(self) -> List[str]:
        return ["input_ids", "attention_mask", "labels"]

    @overrides
    def get_collate_fn(self, state: ExperimentStage) -> Callable:
        if self.is_decoder_only:
            if state == ExperimentStage.TRAINING:
                collator = MyDataCollatorForSeq2Seq(
                    self.tokenizer, label_pad_token_id=-100, padding="longest"
                )
            else:
                collator = DataCollatorForSeq2SeqInCausalLM(
                    self.tokenizer, label_pad_token_id=-100, padding="longest"
                )
        else:
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

            if np.all(preds[:, 0] == tokenizer.pad_token_id):
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

    @overrides
    def get_compute_metric_fn(
        self, stage: ExperimentStage = ExperimentStage.PREDICTION
    ) -> Callable:
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


@dataclass
class DataCollatorForSeq2SeqInCausalLM(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        orig_padding_side = self.tokenizer.padding_side

        if return_tensors is None:
            return_tensors = self.return_tensors

        labels = (
            [feature.pop("labels") for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        prompt_lengths = [len(feature["input_ids"]) for feature in features]

        self.tokenizer.padding_side = "left"
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        input_ids = features["input_ids"]

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            prompt_labels = [l[:p_len] for l, p_len in zip(labels, prompt_lengths)]
            max_prompt_label_length = max(len(l) for l in prompt_labels)

            targets = [l[p_len:] for l, p_len in zip(labels, prompt_lengths)]
            max_label_length = max(len(l) for l in targets)

            padded_targets = []
            # padding_side = "right"
            for tgt in targets:
                remainder = [self.label_pad_token_id] * (max_label_length - len(tgt))
                if isinstance(tgt, list):
                    padded_tgt = tgt + remainder
                else:
                    padded_tgt = np.concatenate([tgt, remainder]).astype(np.int64)

                padded_targets.append(padded_tgt)

            # padding_side = "left"
            padded_prompt_labels = []
            for pr_lbl in prompt_labels:
                remainder = [self.label_pad_token_id] * (
                    max_prompt_label_length - len(pr_lbl)
                )
                if isinstance(pr_lbl, list):
                    padded_pr_lbl = remainder + pr_lbl
                else:
                    padded_pr_lbl = np.concatenate([remainder, pr_lbl]).astype(np.int64)

                padded_prompt_labels.append(padded_pr_lbl)

            padded_targets = torch.tensor(
                padded_targets, dtype=input_ids.dtype
            )
            padded_prompt_labels = torch.tensor(
                padded_prompt_labels, dtype=input_ids.dtype
            )

            labels = torch.cat(
                [padded_prompt_labels, padded_targets],
                dim=1,
            )
            features["labels"] = labels

        self.tokenizer.padding_side = orig_padding_side

        return features


@dataclass
class MyDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    def __call__(self, features, return_tensors=None):
        import numpy as np

        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        self.tokenizer.padding_side = old_padding_side

        return features


if __name__ == "__main__":
    dl_factory = Seq2SeqDataLoaderFactory.from_params(
        Params(
            {
                "data_root": "data",
                "name": "scan",
                "split": "simple",
                "is_decoder_only": True,
                "decoder_only_group_samples": False,
                "decoder_only_block_size": 128,
                "decoder_only_mask_inputs": False,
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

    stage = ExperimentStage.TEST
    ds = dl_factory.get_dataset(stage)
    # ds = ds.with_format(type="pt", columns=["input_ids", "attention_mask", "labels"])
    print(ds)

    ds = ds.remove_columns(
        [
            c
            for c in ds.column_names
            if c not in ["input_ids", "labels", "attention_mask"]
        ]
    )
    dc = dl_factory.get_collate_fn(stage)

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

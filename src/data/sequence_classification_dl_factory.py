import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List

import numpy as np
import wandb
from datasets import Dataset
from overrides import overrides
from transformers import (
    DataCollatorForSeq2Seq,
    EvalPrediction,
    DataCollatorWithPadding,
)

from common import ExperimentStage
from common import Lazy, Params
from data import Seq2SeqDataLoaderFactory
from data.base_dl_factory import DataLoaderFactory
from tokenization_utils import Tokenizer

logger = logging.getLogger("app")


@DataLoaderFactory.register("seq_classification", exist_ok=True)
class SequenceClassificationDataLoaderFactory(Seq2SeqDataLoaderFactory):
    def __init__(
        self,
        input_prompt: Optional[str] = None,
        label_list: Optional[List[str]] = None,
        is_regression: Optional[bool] = False,
        decoder_only_cls_token: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.label_list = label_list
        self.is_regression = is_regression
        self.input_prompt = input_prompt
        self.decoder_only_cls_token = decoder_only_cls_token

        if is_regression and label_list is not None:
            raise ValueError(
                "Cannot have both is_regression and label_list at the same time."
            )

    @property
    def label2id(self) -> Optional[Dict[str, int]]:
        if self.is_regression:
            return None

        return {lbl: idx for idx, lbl in enumerate(self.label_list)}

    def set_tokenizer(self, tokenizer: Tokenizer):
        super().set_tokenizer(tokenizer)
        self.tokenizer.padding_side = "right"

    def get_problem_type(self) -> Optional[str]:
        if self.is_regression:
            return "regression"
        return "single_label_classification"

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

        return ds

    def _get_tokenize_function_for_decoder_only(
        self, add_special_tokens: bool = True, is_training: bool = True
    ) -> Callable:
        tokenizer = self.tokenizer
        max_source_length = self.max_source_length
        label_key = self.target_seq_key

        input_keys = self.input_prompt.split("|")
        decoder_only_input_output_sep_token = self.decoder_only_input_output_sep_token
        decoder_only_cls_token = (
            tokenizer.eos_token
            if self.decoder_only_cls_token is None
            else self.decoder_only_cls_token
        )
        label2id = self.label2id

        def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
            inputs = [example[key] for key in input_keys]
            prompt = (
                decoder_only_input_output_sep_token.join(inputs)
                + f"{decoder_only_cls_token}"
            )

            input_ids = tokenizer(
                prompt,
                truncation=True,
                add_special_tokens=False,
                max_length=max_source_length,
            ).input_ids

            labels = example[label_key]
            if isinstance(labels, str):
                labels = label2id[labels]

            attention_mask = [1] * len(input_ids)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        return tokenize

    def _log_examples(self, ds: Dataset, name: str):
        table = wandb.Table(
            columns=[
                "idx",
                "input_ids",
                "label_id",
                "input_id_tokens",
                "src",
                "label_txt",
            ]
        )
        examples = ds[:100]

        input_keys = self.input_prompt.split("|")

        for idx in range(len(examples[list(examples.keys())[0]])):
            input_ids = [tid for tid in examples["input_ids"][idx] if tid >= 0]
            labels = examples["labels"][idx]

            src = [examples[key][idx] for key in input_keys]

            decoded_input_ids = self.tokenizer.decode(input_ids)
            decoded_input_ids = decoded_input_ids.replace("\n", "\\n")

            input_id_tokens = [
                self.tokenizer.convert_ids_to_tokens(tid) if tid >= 0 else str(tid)
                for tid in examples["input_ids"][idx]
            ]

            label_txt = self.id2label[labels] if not self.is_regression else ""

            table.add_data(
                idx,
                decoded_input_ids,
                labels,
                ", ".join(input_id_tokens),
                src,
                label_txt,
            )

        if wandb.run is not None:
            wandb.log({f"ds_examples/{name}": table})

    @overrides
    def get_collate_fn(self, state: ExperimentStage) -> Callable:
        if self.is_decoder_only:
            collator = DataCollatorForSeqClassificationInCausalLM(
                self.tokenizer,
                padding="longest",
                include_position_ids=self.decoder_only_include_position_ids,
            )
        else:
            collator = DataCollatorForSeq2Seq(
                self.tokenizer, label_pad_token_id=-100, padding="longest"
            )

        return collator

    def get_compute_metric_fn_for_train(
        self,
    ) -> Callable:
        is_regression = self.is_regression

        def compute_metrics(p: EvalPrediction):
            preds = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

            if is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {
                    "accuracy": (preds == p.label_ids).astype(np.float32).mean().item()
                }

        return compute_metrics

    @overrides
    def get_compute_metric_fn(
        self, stage: ExperimentStage = ExperimentStage.PREDICTION
    ) -> Callable:
        return self.get_compute_metric_fn_for_train()


@dataclass
class DataCollatorForSeqClassificationInCausalLM(DataCollatorWithPadding):
    padding_side: str = "right"
    include_position_ids: bool = False

    def __call__(self, features, return_tensors=None):
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = self.padding_side

        outputs = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in outputs:
            outputs["labels"] = outputs["label"]
            del outputs["label"]
        if "label_ids" in outputs:
            outputs["labels"] = outputs["label_ids"]
            del outputs["label_ids"]

        if self.include_position_ids:
            attention_mask = outputs["attention_mask"]
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs["position_ids"] = position_ids

        self.tokenizer.padding_side = orig_padding_side
        return outputs


if __name__ == "__main__":
    dl_factory = SequenceClassificationDataLoaderFactory.from_params(
        Params(
            {
                "data_root": "data",
                "name": "glue_mrpc",
                "split": "normal",
                "label_list": ["not_equivalent", "equivalent"],
                "input_prompt": "sentence1|sentence2",
                "is_regression": False,
                "is_decoder_only": True,
                "train_filename": "train.jsonl",
                "validation_filename": "validation.jsonl",
                "test_filename": "test.jsonl",
                "target_seq_key": "label",
                "decoder_only_input_output_sep_token": "\n",
            }
        )
    )
    tokenizer = Lazy(
        Tokenizer,
        params=Params(
            {
                "type": "pretrained",
                "hf_model_name": "EleutherAI/gpt-neo-125M",
                # "use_fast": False,
                # "type": "whitespace",
            }
        ),
    ).construct(dataset=dl_factory, experiment_root="experiments/base")

    dl_factory.set_tokenizer(tokenizer)

    stage = ExperimentStage.TRAINING
    ds = dl_factory.get_dataset(stage)
    print(ds)

    ds = ds.remove_columns(
        [
            c
            for c in ds.column_names
            if c not in ["input_ids", "labels", "attention_mask", "position_ids"]
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

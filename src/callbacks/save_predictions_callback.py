import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainerState,
    PreTrainedTokenizer,
    TrainerControl,
    TrainingArguments,
)
from wandb.sdk.wandb_run import Run

from callbacks.base_callback import Callback
from common.py_utils import chunks


@Callback.register("save_predictions")
class SavePredictionsCallback(Callback):
    def __init__(self, upload_predictions: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.upload_predictions = upload_predictions

    def init(self, runtime, eval_dataset: Dataset, eval_split: str, **kwargs):
        super().init(runtime, eval_dataset, eval_split, **kwargs)

        from runtime.seq2seq_runtime import Seq2SeqRuntime
        runtime: Seq2SeqRuntime

        self.predictions_dir = runtime.exp_root / f"eval_on_{eval_split}_predictions"
        self.predictions_dir.mkdir(exist_ok=True, parents=True)

        self.dataset = eval_dataset
        self._trainer: Trainer = None
        self._log_counts = 0

    def set_trainer(self, trainer: Trainer):
        self._trainer = trainer

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        tokenizer: PreTrainedTokenizer = None,
        **kwargs,
    ):
        self._save_predictions(state, tokenizer)

    def _save_predictions(self, state: TrainerState, tokenizer: PreTrainedTokenizer):
        test_results = self._trainer.predict(self.dataset, metric_key_prefix=f"pred")

        if state.is_world_process_zero:
            preds = test_results.predictions
            if isinstance(test_results.predictions, tuple):
                preds = preds[0]

            if len(preds.shape) == 3:
                preds = np.argmax(preds, axis=-1)

            output_test_preds_file = (
                self.predictions_dir
                / f"{self._log_counts}_epoch-{str(state.epoch).zfill(5)}_step-{str(state.global_step).zfill(6)}.jsonl"
            )

            with output_test_preds_file.open("w") as writer:
                all_objs = []
                for batch_preds in tqdm(
                    chunks(preds, 128),
                    total=len(preds) // 128,
                    desc="Decoding predictions",
                ):
                    pred_texts = tokenizer.batch_decode(
                        batch_preds,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    pred_texts = [pred.strip() for pred in pred_texts]

                    for pt in pred_texts:
                        all_objs.append({"prediction": pt})

                import jsonlines

                jsonlines.Writer(writer).write_all(all_objs)

        self._log_counts += 1


    def save_outputs(self, logger: Run):
        if self.upload_predictions:
            logger.save("*.jsonl", base_path=self.predictions_dir, policy="now")

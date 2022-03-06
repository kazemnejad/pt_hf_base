from collections import defaultdict, deque
from logging import getLogger

import jsonlines
import wandb
from tqdm import tqdm

from analyzers import Analyzer
from common import ExperimentStage

logger = getLogger("app")


@Analyzer.register("addition_evaluated_acc")
class AdditionEvaluatedAccuracyAnalyzer(Analyzer):
    def __init__(
        self,
        split: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.split = split

    def analyze(self):
        predictions_path = self.exp_root / f"pred_out_{self.split}.jsonl"
        assert (
            predictions_path.exists()
        ), f"Prediction file not found: {predictions_path}"

        pred_objs = []
        with jsonlines.open(str(predictions_path)) as reader:
            for obj in reader:
                pred_objs.append(obj)

        ds_path = self.dl_factory.get_ds_file_path(
            ExperimentStage.from_split(self.split)
        )
        logger.info(f"Evaluating against split: {self.split} at {ds_path}")
        dataset_objs = []
        with jsonlines.open(str(ds_path)) as reader:
            for obj in reader:
                dataset_objs.append(obj)

        assert len(dataset_objs) == len(pred_objs)

        from data.addition_task_dl_factory import SCRATCHPAD_SEP

        accuracies = defaultdict(deque)

        evaluation_table = wandb.Table(
            columns=["idx", "is_eval_correct", "parse_error"]
        )
        for idx, (pred_obj, ds_obj) in tqdm(
            enumerate(zip(pred_objs, dataset_objs)), total=len(pred_objs)
        ):
            source = ds_obj["source"]
            target = ds_obj["target"]
            category = ds_obj["_gen_src"]
            prediction = pred_obj["prediction"]

            source = source.replace(" ", "")
            target = target.strip().replace(" ", "")

            truth = sum(int(x) for x in source.split("+"))  # get the sum

            assert int(target) == truth

            exp = ""
            try:
                # get the model's answer
                # last line of output is model's answer
                prediction = prediction.split(SCRATCHPAD_SEP)[-1]
                prediction = int(prediction.replace(" ", ""))
                is_correct = int(prediction == truth)
            except Exception as exp:
                logger.warning(f"Could parse the model's prediction {exp}")
                is_correct = 0
                exp = str(exp)

            accuracies[category].append(is_correct)

            evaluation_table.add_data(idx, bool(is_correct), exp)

        self.logger.log({f"evaluated_acc/{self.split}_table": evaluation_table})

        stats = []
        for key, acc_lst in accuracies.items():
            acc = sum(acc_lst) / len(acc_lst)
            acc = round(acc, 4)
            stats.append((f"{key}", acc))
            self.logger.log({f"pred/{self.split}_eAcc_{key}": acc})

        all_predictions = [
            is_correct for acc_lst in accuracies.values() for is_correct in acc_lst
        ]
        overall_acc = sum(all_predictions) / len(all_predictions)
        overall_acc = round(overall_acc, 4)
        stats.append(("overall", overall_acc))

        plot = wandb.plot.bar(
            wandb.Table(data=stats, columns=["split", "eAcc"]),
            label="split",
            value="eAcc",
            title=f"Evaluated accuracy (eAcc) in split: {self.split}",
        )

        self.logger.log({f"evaluated_acc/{self.split}": plot})

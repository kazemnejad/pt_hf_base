from datasets import load_metric
from transformers import AutoModel, AutoTokenizer

model_names = [
    "roberta-base",
    "roberta-large",
    "albert-base-v2",
    "albert-large-v2",
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "t5-small",
    "t5-base",
    "t5-large",
    "EleutherAI/gpt-neo-125M",
    "bert-base-uncased",
    "bert-base-cased",
]
for model_name in model_names:
    AutoTokenizer.from_pretrained(model_name)
for model_name in model_names:
    AutoModel.from_pretrained(model_name)

load_metric("glue", "rte")
load_metric("accuracy")
load_metric("seqeval")
load_metric("squad")

from transformers import Seq2SeqTrainer

from common import Registrable


class BaseTrainer(Seq2SeqTrainer, Registrable):
    pass

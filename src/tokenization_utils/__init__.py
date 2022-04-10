from typing import List


class SpecialTokens:
    PAD = "<pad>"
    START = "<s>"
    END = "</s>"
    UNK = "<unk>"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.PAD, cls.END, cls.UNK, cls.START]

from .base_tokenizer import Tokenizer
from .pre_trained import DIPreTrainedTokenizer
from .whitespace import WhitespaceTokenizer
from .manual import ManualWhitespaceTokenizer
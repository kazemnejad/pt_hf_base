import logging
from typing import Optional

from transformers import (
    T5ForConditionalGeneration,
    T5Config,
)

from models.base_model import Model, HfModelConfig
from tokenization_utils import Tokenizer

logger = logging.getLogger("app")


@HfModelConfig.register("seq2seq_t5", "from_di")
class DiT5Config(T5Config, HfModelConfig):
    pass


@Model.register("seq2seq_t5")
class Seq2SeqT5(T5ForConditionalGeneration, Model):
    def __init__(
        self,
        config: Optional[HfModelConfig] = None,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs,
    ):
        assert config is not None
        super().__init__(config)

        # For some reason T5's number of embeddings > vocab_size,
        # and apparently those extra embedding are not used by default.
        # So, we can use them for our re-sized vocab_size
        if (
            tokenizer is not None
            and len(tokenizer) > self.shared.num_embeddings
            or len(tokenizer) < config.vocab_size
        ):
            logger.info(
                f"Resizing num_embeddings to {len(tokenizer)} (Previously, {self.shared.num_embeddings})"
            )
            self.resize_token_embeddings(len(tokenizer))


if __name__ == "__main__":
    from common import Params

    # DiT5Config.default_implementation = "t5"
    model = HfModelConfig.from_params(
        Params(
            {
                "type": "t5",
                "hf_model_name": "t5-large",
            }
        )
    )
    print(model)

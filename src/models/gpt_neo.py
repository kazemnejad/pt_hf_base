import logging
from typing import Optional

from transformers import (
    GPTNeoConfig,
)
from transformers import GPTNeoForCausalLM

from models.base_model import Model, HfModelConfig
from tokenization_utils import Tokenizer

logger = logging.getLogger("app")


@HfModelConfig.register("gpt_neo", "from_di")
class DiGPTNeoConfig(GPTNeoConfig, HfModelConfig):
    pass


@Model.register("gpt_neo")
class CausalGPTNeo(GPTNeoForCausalLM, Model):
    def __init__(
        self,
        config: Optional[HfModelConfig] = None,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs,
    ):
        assert config is not None
        super().__init__(config)

        self.tokenizer = tokenizer

    def post_checkpoint_load(self):
        tokenizer = self.tokenizer
        if tokenizer is not None:
            if (
                len(tokenizer) > self.transformer.wte.num_embeddings
                or len(tokenizer) < self.config.vocab_size
            ):
                logger.info(
                    f"Resizing num_embeddings to {len(tokenizer)} (Previously, {self.transformer.wte.num_embeddings})"
                )
                self.resize_token_embeddings(len(tokenizer))

        del self.tokenizer


if __name__ == "__main__":
    from common import Params

    # DiT5Config.default_implementation = "t5"
    model = Model.from_params(
        Params(
            {
                "type": "gpt_neo",
                "config": {
                    "type": "gpt_neo",
                    "hf_model_name": "EleutherAI/gpt-neo-125M",
                },
                "hf_model_name": "EleutherAI/gpt-neo-125M",
            }
        )
    )
    print(model)

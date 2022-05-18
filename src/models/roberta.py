import logging
from typing import Optional, Dict

from transformers import RobertaConfig, RobertaForSequenceClassification

from models.base_model import Model, HfModelConfig
from tokenization_utils import Tokenizer

logger = logging.getLogger("app")


@HfModelConfig.register("roberta", "from_di")
class DiRobertaConfig(RobertaConfig, HfModelConfig):
    pass


@Model.register("roberta_seq_classifier")
class SeqClassifierRoberta(RobertaForSequenceClassification, Model):
    def __init__(
        self,
        config: Optional[HfModelConfig] = None,
        tokenizer: Optional[Tokenizer] = None,
        problem_type: Optional[str] = "single_label_classification",
        label2id: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        assert config is not None

        if problem_type is not None:
            config.problem_type = problem_type
        if label2id is not None:
            config.label2id = label2id
            config.id2label = {idx: lbl for lbl, idx in label2id.items()}

        super().__init__(config)

        self.handle_tokenizer(tokenizer)

    def handle_tokenizer(self, tokenizer: Optional[Tokenizer] = None):
        if tokenizer is None:
            return

        self.config.eos_token_id = tokenizer.eos_token_id
        self.config.bos_token_id = tokenizer.bos_token_id
        self.config.pad_token_id = tokenizer.pad_token_id

        embedding = self.roberta.get_input_embeddings()

        if (
            len(tokenizer) > embedding.num_embeddings
            or len(tokenizer) < self.config.vocab_size
        ):
            logger.info(
                f"Resizing num_embeddings to {len(tokenizer)} (Previously, {embedding.num_embeddings})"
            )
            self.resize_token_embeddings(len(tokenizer))


if __name__ == "__main__":
    from common import Params

    model = Model.from_params(
        Params(
            {
                "type": "roberta_seq_classifier",
                "config": {
                    "type": "roberta",
                    "return_dict": True,
                    "output_hidden_states": False,
                    "output_attentions": False,
                    "torchscript": False,
                    "torch_dtype": None,
                    "use_bfloat16": False,
                    "pruned_heads": {},
                    "tie_word_embeddings": True,
                    "is_encoder_decoder": False,
                    "is_decoder": False,
                    "cross_attention_hidden_size": None,
                    "add_cross_attention": False,
                    "tie_encoder_decoder": False,
                    "max_length": 20,
                    "min_length": 0,
                    "do_sample": False,
                    "early_stopping": False,
                    "num_beams": 1,
                    "num_beam_groups": 1,
                    "diversity_penalty": 0.0,
                    "temperature": 1.0,
                    "top_k": 50,
                    "top_p": 1.0,
                    "repetition_penalty": 1.0,
                    "length_penalty": 1.0,
                    "no_repeat_ngram_size": 0,
                    "encoder_no_repeat_ngram_size": 0,
                    "bad_words_ids": None,
                    "num_return_sequences": 1,
                    "chunk_size_feed_forward": 0,
                    "output_scores": False,
                    "return_dict_in_generate": False,
                    "forced_bos_token_id": None,
                    "forced_eos_token_id": None,
                    "remove_invalid_values": False,
                    "architectures": ["RobertaForMaskedLM"],
                    "finetuning_task": None,
                    "id2label": {0: "LABEL_0", 1: "LABEL_1"},
                    "label2id": {"LABEL_0": 0, "LABEL_1": 1},
                    "tokenizer_class": None,
                    "prefix": None,
                    "bos_token_id": 0,
                    "pad_token_id": 1,
                    "eos_token_id": 2,
                    "sep_token_id": None,
                    "decoder_start_token_id": None,
                    "task_specific_params": None,
                    "problem_type": None,
                    "_name_or_path": "roberta-base",
                    "transformers_version": "4.16.2",
                    "model_type": "roberta",
                    "vocab_size": 50265,
                    "hidden_size": 768,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 12,
                    "hidden_act": "gelu",
                    "intermediate_size": 3072,
                    "hidden_dropout_prob": 0.1,
                    "attention_probs_dropout_prob": 0.1,
                    "max_position_embeddings": 514,
                    "type_vocab_size": 1,
                    "initializer_range": 0.02,
                    "layer_norm_eps": 1e-05,
                    "position_embedding_type": "absolute",
                    "use_cache": True,
                    "classifier_dropout": None,
                },
            }
        )
    )
    print(model)

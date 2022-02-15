from typing import List, Optional
from torch import nn
import torch
from transformers.modeling_utils import PreTrainedModel

GLOBAL_PROMPT_KEY = "_global"

class PromptWrapper(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        prompt_length: int = 10,
        domain_prompt_length: int = 0,
        domains: List[str] = [],
        random_range: float = 0.5,
        initialize_from_vocab: bool = True,
    ):
        super(PromptWrapper, self).__init__()

        self.prompt_length = prompt_length
        self.domain_prompt_length = domain_prompt_length
        self.model = model
        self._prompts = nn.ParameterDict()
        self._prompt_init = lambda length: self.initialize_embedding(
            model.get_input_embeddings(), length, random_range, initialize_from_vocab
        )

        # intitialize parameters
        self.get_prompt(GLOBAL_PROMPT_KEY, prompt_length)

        for domain in domains:
            self.get_prompt(domain, domain_prompt_length)

    def get_prompt(self, domain, length=None):
        if domain in self._prompts:
            return self._prompts[domain]

        params = nn.parameter.Parameter(self._prompt_init(length or self.domain_prompt_length))
        self._prompts[domain] = params
        return params

    def initialize_embedding(
        self,
        embedding: nn.Embedding,
        prompt_length: int = 10,
        random_range: float = 0.5,
        initialize_from_vocab: bool = True,
    ):
        if initialize_from_vocab:
            return embedding.weight[:prompt_length].clone().detach()
        return torch.FloatTensor(prompt_length, embedding.weight.size(1)).uniform_(-random_range, random_range)

    def build_inputs(self, input_ids, attention_mask, labels=None, domains=None):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        prompt_length = self.prompt_length + (self.domain_prompt_length if domains else 0)
        if prompt_length and attention_mask is not None:
            padding = torch.full((batch_size, prompt_length), 1).to(device)
            attention_mask = torch.cat((padding, attention_mask), dim=1)

        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if prompt_length:
            prompt = self.get_prompt(GLOBAL_PROMPT_KEY).repeat(batch_size, 1, 1)

            if labels is not None and labels.shape == input_ids.shape:
                # Autoregressive need to pad labels as well
                padding = torch.full((batch_size, prompt_length), -100).to(device)
                labels = torch.cat([padding, labels], dim=1)

            if domains:
                domain_prompt = torch.stack([self.get_prompt(d) for d in domains])
                prompt = torch.cat([prompt, domain_prompt], dim=1)

            inputs_embeds = torch.cat([prompt, inputs_embeds], 1)
        return inputs_embeds, attention_mask, labels

    def forward(self, input_ids, attention_mask, domains=[], labels=None, **kwargs):
        inputs_embeds, attention_mask, labels = self.build_inputs(input_ids, attention_mask, labels, domains)

        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)

    def generate(self, input_ids=None, attention_mask=None, domains=[], **kwargs):
        inputs_embeds, attention_mask, labels = self.build_inputs(input_ids, attention_mask, labels=None, domains=domains)
        max_length = kwargs.pop("max_length", None)
        max_length = max_length + 10 if max_length else None

        if self.model.config.is_encoder_decoder:
            model_kwargs = { 'encoder_outputs': self.model.get_encoder()(inputs_embeds=inputs_embeds) }
        else:
            model_kwargs = { 'inputs_embeds': inputs_embeds }

        return self.model.generate(
            input_ids=None,
            use_cache=True,
            no_repeat_ngram_size=0,
            max_length=max_length,
            **model_kwargs,
            **kwargs,
        )
    
    @property
    def config(self):
        return self.model.config
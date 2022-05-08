import logging
import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers import T5Config
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.models.t5.modeling_t5 import (
    T5Stack,
    T5PreTrainedModel,
    T5Block,
    T5LayerNorm,
    T5LayerFF,
    T5LayerSelfAttention,
    T5Attention,
    T5LayerCrossAttention,
)
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

from models.base_model import Model, HfModelConfig
from tokenization_utils import Tokenizer

logger = logging.getLogger("app")

POSITION_ENCODING_REL_T5_BIAS = "t5_relative_bias"
POSITION_ENCODING_REL_TRANSFORMER_XL = "transformer_xl_relative_encoding"
POSITION_ENCODING_ROTARY = "rotary"
POSITION_ENCODING_ABS_LEARNED = "abs_learned"
POSITION_ENCODING_ABS_SINUSOID = "abs_sinusoid"
POSITION_ENCODING_ALiBi = "alibi"
POSITION_ENCODING_ALiBi_LEARNED = "alibi_learned"
POSITION_ENCODING_NONE = "none"


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq)
        .to(x.device)
        .float()
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(
        lambda t: t[None, offset : x.shape[1] + offset, None, :].repeat_interleave(
            2, 3
        ),
        sincos,
    )
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


class FixedAbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(2048).type_as(inv_freq)
        sinusoid_inp = torch.einsum("i , j -> i j", t, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.embed = nn.Embedding.from_pretrained(emb, freeze=True)

    def forward(self, position_ids: torch.Tensor):
        return self.embed(position_ids.long())


class CustomT5Attention(T5Attention):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super(T5Attention, self).__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.d_head = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.position_encoding_type = getattr(
            config, "position_encoding_type", POSITION_ENCODING_REL_T5_BIAS
        )

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )

        if self.position_encoding_type == POSITION_ENCODING_REL_TRANSFORMER_XL:
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_heads, self.d_head))
            nn.init.normal_(
                self.r_r_bias, mean=0.0, std=config.initializer_factor * 0.2
            )
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_heads, self.d_head))
            nn.init.normal_(
                self.r_w_bias, mean=0.0, std=config.initializer_factor * 0.2
            )
            self.r = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
            self.r.weight.data.normal_(
                mean=0.0, std=config.initializer_factor * (self.d_model**-0.5)
            )
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.clamp_length = 1000

        if self.position_encoding_type == POSITION_ENCODING_ROTARY:
            self.rotary_dim = None
            if getattr(config, "rotary_dim", None) is not None:
                self.rotary_dim = config.rotary_dim
            self.rotary_dim = 4

        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def _rel_shift(self, x):
        zero_pad_shape = x.size()[:2] + (x.size(2), 1)
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)
        x_padded_shape = x.size()[:2] + (x.size(3) + 1, x.size(2))
        x_padded = x_padded.view(*x_padded_shape)
        x = x_padded[:, :, 1:, :].view_as(x)
        return x

    def forward(
        self,
        hidden_states,
        mask=None,
        position_bias=None,
        key_value_states=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        key_length = (
            real_seq_length if key_value_states is None else key_value_states.shape[1]
        )

        def shape(states):
            """projection"""
            return states.view(
                batch_size, -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return (
                states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
            )

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        if self.position_encoding_type == POSITION_ENCODING_ROTARY:
            key_states = shape(self.k(hidden_states))
        else:
            # get key/value states
            key_states = project(
                hidden_states,
                self.k,
                key_value_states,
                past_key_value[0] if past_key_value is not None else None,
            )

        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        if self.position_encoding_type == POSITION_ENCODING_REL_T5_BIAS:
            scores = torch.matmul(query_states, key_states.transpose(3, 2))
            if position_bias is None:
                if not self.has_relative_attention_bias:
                    position_bias = torch.zeros(
                        (1, self.n_heads, real_seq_length, key_length),
                        device=scores.device,
                        dtype=scores.dtype,
                    )
                    if self.gradient_checkpointing and self.training:
                        position_bias.requires_grad = True
                else:
                    position_bias = self.compute_bias(real_seq_length, key_length)

                # if key and values are already calculated
                # we want only the last query position bias
                if past_key_value is not None:
                    position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

                if mask is not None:
                    position_bias = (
                        position_bias + mask
                    )  # (batch_size, n_heads, seq_length, key_length)

            scores += position_bias
        elif self.position_encoding_type == POSITION_ENCODING_REL_TRANSFORMER_XL:
            if position_bias is None:
                pos_seq = torch.arange(
                    real_seq_length - 1,
                    -1,
                    -1.0,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                if self.clamp_length > 0:
                    pos_seq = pos_seq.clamp_(max=self.clamp_length)
                position_bias = self.pos_emb(pos_seq)
                position_bias = nn.functional.dropout(
                    position_bias, p=self.dropout, training=self.training
                )

            position_embeds = position_bias  # position embeds: [1, seq_len, d_model]

            r_head_k = self.r(position_embeds)  # [1, seq_len, n_head*d_head]
            r_head_k = r_head_k.view(
                position_embeds.shape[1], self.n_heads, self.d_head
            )  # [seq_len, n_head, d_head]

            rw_head_q = query_states + self.r_w_bias[None, :, None, :]
            AC = torch.einsum("bnqd,bnkd->bnqk", (rw_head_q, key_states))

            rr_head_q = query_states + self.r_r_bias[None, :, None, :]
            BD = torch.einsum("bnid,jnd->bnij", (rr_head_q, r_head_k))
            BD = self._rel_shift(BD)

            scores = AC + BD

            if mask is not None:
                scores += mask
        elif self.position_encoding_type == POSITION_ENCODING_ROTARY:
            r_seq_len = hidden_states.shape[1]
            r_offset = 0

            if past_key_value is not None:
                r_offset = past_key_value[0].shape[2]
                r_seq_len += r_offset

            query_states = query_states.permute(0, 2, 1, 3)
            key_states = key_states.permute(0, 2, 1, 3)

            if self.rotary_dim is not None:

                k_rot = key_states[:, :, :, : self.rotary_dim]
                k_pass = key_states[:, :, :, self.rotary_dim :]

                q_rot = query_states[:, :, :, : self.rotary_dim]
                q_pass = query_states[:, :, :, self.rotary_dim :]

                sincos = fixed_pos_embedding(k_rot, 1, seq_len=r_seq_len)
                k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=r_offset)
                q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=r_offset)

                key_states = torch.cat([k_rot, k_pass], dim=-1)
                query_states = torch.cat([q_rot, q_pass], dim=-1)
            else:
                sincos = fixed_pos_embedding(key_states, 1, seq_len=r_seq_len)
                key_states = apply_rotary_pos_emb(key_states, sincos, offset=r_offset)
                query_states = apply_rotary_pos_emb(
                    query_states, sincos, offset=r_offset
                )

            query_states = query_states.permute(0, 2, 1, 3)
            key_states = key_states.permute(0, 2, 1, 3)

            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)

            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
            if mask is not None:
                scores += mask  # (batch_size, n_heads, seq_length, key_length)
        else:
            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
            if mask is not None:
                scores += mask  # (batch_size, n_heads, seq_length, key_length)

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(
            torch.matmul(attn_weights, value_states)
        )  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class CustomT5LayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super(T5LayerSelfAttention, self).__init__()
        self.SelfAttention = CustomT5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)


class CustomT5Block(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super(T5Block, self).__init__()
        self.is_decoder = config.is_decoder
        assert self.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            CustomT5LayerSelfAttention(
                config, has_relative_attention_bias=has_relative_attention_bias
            )
        )
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))


class CustomT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super(T5Stack, self).__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.position_encoding_type = getattr(
            config, "position_encoding_type", POSITION_ENCODING_REL_T5_BIAS
        )

        logger.info(f"position_encoding_type: {self.position_encoding_type}")

        self.block = nn.ModuleList(
            [
                CustomT5Block(config, has_relative_attention_bias=bool(i == 0))
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        if self.position_encoding_type == POSITION_ENCODING_ABS_LEARNED:
            self.wpe = nn.Embedding(2048, config.d_model)
            self.wpe.weight.data.normal_(mean=0.0, std=config.initializer_factor * 1.0)

        if self.position_encoding_type == POSITION_ENCODING_ABS_SINUSOID:
            self.wpe = FixedAbsolutePositionalEmbedding(config.d_model)

        if self.position_encoding_type in [
            POSITION_ENCODING_ALiBi,
            POSITION_ENCODING_ALiBi_LEARNED,
        ]:
            maxpos = 2048
            attn_heads = config.num_heads
            if self.position_encoding_type == POSITION_ENCODING_ALiBi_LEARNED:
                self.learned_logslopes = nn.Parameter(
                    torch.log(torch.Tensor(self.get_slopes(attn_heads)))
                )
            else:
                slopes = torch.Tensor(self.get_slopes(attn_heads))
                alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos).unsqueeze(
                    0
                ).unsqueeze(0).expand(attn_heads, -1, -1)
                alibi = alibi.view(attn_heads, 1, maxpos)
                self.register_buffer("alibi", alibi)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(
                n
            )  # In the paper, we only train models that have 2^a heads for some a. This function has
        else:  # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2 ** math.floor(
                math.log2(n)
            )  # when the number of heads is not a power of 2, we use this workaround.
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self.get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        position_ids=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds"
            )

        if inputs_embeds is None:
            assert (
                self.embed_tokens is not None
            ), "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        if self.position_encoding_type in [
            POSITION_ENCODING_ABS_LEARNED,
            POSITION_ENCODING_ABS_SINUSOID,
        ]:
            if position_ids is not None:
                position_ids = position_ids.view(-1, input_shape[-1])

            if past_key_values is None:
                past_length = 0
            else:
                past_length = past_key_values[0][0].size(-2)

            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if position_ids is None:
                position_ids = torch.arange(
                    past_length,
                    input_shape[-1] + past_length,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

            position_embeds = self.wpe(position_ids)
            inputs_embeds += position_embeds

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            past_key_values[0][0].shape[2] + seq_length
            if past_key_values is not None
            else seq_length
        )

        if use_cache is True:
            assert (
                self.is_decoder
            ), f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(
                inputs_embeds.device
            )
        if (
            self.is_decoder
            and encoder_attention_mask is None
            and encoder_hidden_states is not None
        ):
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size,
                encoder_seq_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, inputs_embeds.device
        )

        if self.position_encoding_type in [
            POSITION_ENCODING_ALiBi,
            POSITION_ENCODING_ALiBi_LEARNED,
        ]:
            if not hasattr(self, "alibi"):
                maxpos = 2048
                attn_heads = self.config.num_heads
                slopes = self.learned_logslopes.exp()
                alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(
                    maxpos, device=slopes.device
                ).unsqueeze(0).unsqueeze(0).expand(attn_heads, -1, -1)
                alibi = alibi.view(attn_heads, 1, maxpos)
            else:
                alibi = self.alibi

            alibi = alibi.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            alibi = alibi[:, :, :, : attention_mask.shape[-1]]
            alibi = alibi.repeat(1, 1, extended_attention_mask.shape[2], 1)
            extended_attention_mask = torch.where(
                extended_attention_mask == 0,
                alibi,
                extended_attention_mask.repeat(1, self.config.num_heads, 1, 1),
            )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(
            cross_attn_head_mask, self.config.num_layers
        )
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(
            zip(self.block, past_key_values)
        ):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(
                        hidden_states.device
                    )
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = (
                        encoder_extended_attention_mask.to(hidden_states.device)
                    )
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(
                        hidden_states.device
                    )
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(
                        hidden_states.device
                    )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    4 if output_attentions else 3
                ]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (
                    present_key_value_state,
                )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


@Model.register("custom_decoder_only_t5")
class CustomDecoderOnlyT5(T5PreTrainedModel, Model):

    _keys_to_ignore_on_load_missing = [
        r"decoder\.embed_tokens\.weight",
        r"encoder",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(
        self,
        config: Optional[HfModelConfig] = None,
        position_encoding_type: Optional[str] = None,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs,
    ):
        assert config is not None
        config.is_decoder = True
        config.is_encoder_decoder = False
        if position_encoding_type is not None:
            if position_encoding_type not in [
                POSITION_ENCODING_ALiBi,
                POSITION_ENCODING_ALiBi_LEARNED,
                POSITION_ENCODING_ABS_LEARNED,
                POSITION_ENCODING_ABS_SINUSOID,
                POSITION_ENCODING_REL_T5_BIAS,
                POSITION_ENCODING_REL_TRANSFORMER_XL,
                POSITION_ENCODING_ROTARY,
                POSITION_ENCODING_NONE,
            ]:
                raise ValueError(
                    f"Invalid position_encoding_type: {position_encoding_type}"
                )
            config.position_encoding_type = position_encoding_type

        self.main_input_name = "input_ids"

        super().__init__(config)

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder = CustomT5Stack(config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        #
        cross_attention_params = [
            p
            for n, p in self.decoder.named_parameters()
            if n.startswith("block.") and ".layer.1." in n
        ]
        for param in cross_attention_params:
            param.requires_grad = False

        self.handle_tokenizer(tokenizer)

    def get_decoder(self):
        return self.decoder

    def handle_tokenizer(self, tokenizer: Optional[Tokenizer] = None):
        if tokenizer is None:
            return

        self.config.eos_token_id = tokenizer.eos_token_id
        self.config.bos_token_id = self.config.eos_token_id

        if (
            len(tokenizer) > self.shared.num_embeddings
            or len(tokenizer) < self.config.vocab_size
        ):
            logger.info(
                f"Resizing num_embeddings to {len(tokenizer)} (Previously, {self.shared.num_embeddings})"
            )
            self.resize_token_embeddings(len(tokenizer))

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            if input_ids is not None:
                input_ids = input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)

        transformer_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            position_ids=position_ids,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=head_mask,
            cross_attn_head_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            hidden_states = hidden_states * (self.model_dim**-0.5)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Compute loss in fp32 to match with mesh-tf version
            # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )


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

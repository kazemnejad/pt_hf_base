local base = (import 'base.jsonnet');

base + {
    model+: {
        type: 'custom_decoder_only_t5',
        hf_model_name: 't5-base',
        config+: {
            type: 'seq2seq_t5',
            architectures: [
                'T5WithLMHeadModel',
            ],
            d_ff: 256,
            d_kv: 8,
            d_model: 64,
            decoder_start_token_id: 0,
            dropout_rate: 0.1,
            eos_token_id: 1,
            feed_forward_proj: 'relu',
            initializer_factor: 1.0,
            is_encoder_decoder: false,
            is_decoder: true,
            layer_norm_epsilon: 1e-06,
            model_type: 't5',
            n_positions: 512,
            num_decoder_layers: 6,
            num_heads: 8,
            num_layers: 6,
            output_past: true,
            pad_token_id: 0,
            relative_attention_num_buckets: 32,
            use_cache: true,
            vocab_size: 32128,
        },
    },
}

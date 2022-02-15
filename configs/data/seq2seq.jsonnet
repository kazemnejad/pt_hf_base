local base = (import 'base.jsonnet');

base + {
    dataset+: {
        type: 'seq2seq',
        source_seq_key: 'source',
        target_seq_key: 'target',
        append_vocab: 'no',
        max_source_length: 256,
        max_target_length: 256,
    },
}

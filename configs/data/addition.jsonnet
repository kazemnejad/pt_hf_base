local base = (import 'seq2seq.jsonnet');

base + {
    dataset+: {
        type: 'addition_task',
        name: 'addition',
        split: 'normal',
        include_scratchpad: false,
        is_seq2seq: false,
        max_source_length: 30,
        max_target_length: 256,
    },
}

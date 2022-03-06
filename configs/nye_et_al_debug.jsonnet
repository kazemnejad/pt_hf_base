(import 'base.jsonnet')
+ (import 'models/gpt_neo_125M.jsonnet')
+ (import 'tokenizers/pretrained.jsonnet')
+ (import 'trainer/nye_et_al.jsonnet')
+ (import 'data/addition.jsonnet')
+ {
    global_vars+: {
        debug_mode: true,
    },
    dataset+: {
        is_seq2seq: false,
        include_scratchpad: false,
    },
    model+: {
        from_pretrained: false
    }
}

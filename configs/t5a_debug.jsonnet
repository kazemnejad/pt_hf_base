(import 'base.jsonnet')
+ (import 'models/t5a_debug.jsonnet')
+ (import 'tokenizers/whitespace.jsonnet')
+ (import 'trainer/debug.jsonnet')
+ {
    global_vars+: {
        debug_mode: true,
    },
    dataset+: {
        num_proc: 1,
    }
}

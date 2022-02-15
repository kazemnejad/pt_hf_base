local base = (import 'base.jsonnet');


local hf_model_name = "t5-small";
local type = "seq2seq_t5";

base + {
    model+: {
        type: type,
        hf_model_name: hf_model_name,
        config+: {
            type: type,
            hf_model_name: hf_model_name,
        }
    },
}

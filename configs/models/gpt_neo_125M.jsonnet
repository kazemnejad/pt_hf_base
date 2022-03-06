local base = (import 'base.jsonnet');


local hf_model_name = 'EleutherAI/gpt-neo-125M';
local type = 'gpt_neo';

base + {
    model+: {
        type: type,
        hf_model_name: hf_model_name,
        config+: {
            type: type,
            hf_model_name: hf_model_name,
        },
        from_pretrained: true,
    },
}

local base = (import 'base.jsonnet');


local hp_max_steps = {
    values: [50 * 1000, 100 * 1000, 200 * 1000],
};
local hp_lr = {
    values: [0.0003, 0.00001, 0.0001],
};

local hp_weight_decay = {
    values: [0.1, 0],
};

local hp_warmup_ratio = {
    values: [0, 0.06],
};

local hp_lr_scheduler_type = {
    values: ['cosine'],
};

base + {
    method: 'grid',
    metric: {
        goal: 'maximize',
        name: 'pred/test_seq_acc',
    },
    parameters+: {
        trainer+: {
            max_steps: std.manifestJsonMinified(hp_max_steps),
            learning_rate: std.manifestJsonMinified(hp_lr),
            weight_decay: std.manifestJsonMinified(hp_weight_decay),
            lr_scheduler_type: std.manifestJsonMinified(hp_lr_scheduler_type),
            warmup_ratio: std.manifestJsonMinified(hp_warmup_ratio)
        },
    },
}

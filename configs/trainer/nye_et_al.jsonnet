local base = (import 'base.jsonnet');

base + {
    trainer+: {
        max_steps: 5000,
        eval_steps: 1000,
        logging_steps: 20,

        save_total_limit: 5,

        per_device_train_batch_size: 32,
        per_device_eval_batch_size: 64,

        generation_max_length: 512,

        learning_rate: 3e-5,

        dataloader_num_workers: 4,
        dataloader_pin_memory: true,
    },
}

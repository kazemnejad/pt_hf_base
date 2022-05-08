{
    dataset+: {
        name: 'scan',
        split: std.extVar("APP_DS_SPLIT"),
        train_filename: 'train.jsonl',
        validation_filename: 'valid.jsonl',
        test_filename: 'test.jsonl',
    },
}

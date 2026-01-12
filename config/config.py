def get_config():
    return {
        "batch_size": 2,
        "num_epochs": 3,
        "lr": 0.0001,
        "seq_length": 128,
        "d_model": 128,
        "nhead": 8,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "vocab_size": 10000,
        "tokenizer_batch_size": 64,
        "source_lang": "en",
        "target_lang": "hi",
        "lang_src": "en",
        "lang_tgt": "hi",
        "model_folder": "models",
        "tokenizer_folder": "data/tokenizers",
        "preload": None,
        "tokenizer_file": "data/tokenizers/tokenizer_{lang}.json",
        "experiment_name": "runs/en_hi_model",
        "max_seq_filter": 500,
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    experiment_name = config['experiment_name']
    model_filename = f"{experiment_name}_epoch{epoch}.pt"
    return f"{model_folder}/{experiment_name}/{model_filename}"
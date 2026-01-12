import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from src.core.dataset import BilingualDataset, casual_mask
from src.core.model import build_transformer
from config.config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentence(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(encoder_input, encoder_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)
    
    while True:
        if decoder_input.size(1) >= max_len:
            break
        
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(encoder_input).to(device)
        decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
        projected_output = model.project(decoder_output)
        
        _, next_word = torch.max(projected_output[:, -1, :], dim=1)
        next_word = next_word.item()
        
        decoder_input = torch.cat(
            [decoder_input,
             torch.empty(1, 1).fill_(next_word).type_as(encoder_input).to(device)],
            dim=1
        )
        
        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)

def run_validation(model, validation_df, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0
    source_txt = []
    expected = []
    predicted = []

    console_width = 80
    
    with torch.no_grad():
        for batch in validation_df:
            count += 1
            
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['enc_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            
            source_txt.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            print_msg('-' * console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")
            print_msg('-' * console_width)

            if count == num_examples:
                break

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang=lang))
    if not tokenizer_path.is_file():
        print(f"Building tokenizer for {lang}...")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            vocab_size=config['vocab_size'],
        )

        def batch_iterator():
            for i in range(0, len(ds), config['tokenizer_batch_size']):
                batch = [ds[j] for j in range(i, min(i + config['tokenizer_batch_size'], len(ds)))]
                yield [str(x) for x in batch]

        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
    else:
        print(f"Loading tokenizer for {lang} from {tokenizer_path}...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('cfilt/iitb-english-hindi', split='train')

    class DatasetWrapper:
        def __init__(self, ds, lang):
            self.ds = ds
            self.lang = lang
        def __getitem__(self, idx):
            return self.ds[idx]['translation'][self.lang]
        def __len__(self):
            return len(self.ds)
    
    src_wrapper = DatasetWrapper(ds_raw, config['source_lang'])
    tgt_wrapper = DatasetWrapper(ds_raw, config['target_lang'])
    tokenizer_src = get_or_build_tokenizer(config, src_wrapper, config['source_lang']) 
    tokenizer_tgt = get_or_build_tokenizer(config, tgt_wrapper, config['target_lang'])

    # FILTER OUT LONG SENTENCES
    print("Filtering dataset for sequence length...")
    max_seq_filter = config.get('max_seq_filter', config['seq_length'] - 20)
    
    filtered_indices = []
    for idx in range(len(ds_raw)):
        item = ds_raw[idx]
        src_text = item['translation'][config['source_lang']]
        tgt_text = item['translation'][config['target_lang']]
        
        enc_tokens = tokenizer_src.encode(src_text).ids
        dec_tokens = tokenizer_tgt.encode(tgt_text).ids
        
        # Keep only if both are within limits (accounting for special tokens)
        if len(enc_tokens) <= max_seq_filter and len(dec_tokens) <= max_seq_filter:
            filtered_indices.append(idx)
    
    print(f"Kept {len(filtered_indices)} / {len(ds_raw)} examples after filtering")
    
    # Select only filtered examples
    ds_filtered = ds_raw.select(filtered_indices)

    # Split into train and validation
    train_ds_raw, val_ds_raw = random_split(
        ds_filtered, 
        [int(0.9 * len(ds_filtered)), len(ds_filtered) - int(0.9 * len(ds_filtered))]
    )
    
    train_ds = BilingualDataset(
        train_ds_raw, tokenizer_src, tokenizer_tgt, 
        config['source_lang'], config['target_lang'], config['seq_length']
    )
    val_ds = BilingualDataset(
        val_ds_raw, tokenizer_src, tokenizer_tgt, 
        config['source_lang'], config['target_lang'], config['seq_length']
    )

    # Calculate statistics on filtered dataset
    max_len_src = 0
    max_len_tgt = 0

    for idx in filtered_indices[:1000]:  # Sample first 1000 for speed
        item = ds_raw[idx]
        src_text = item['translation'][config['source_lang']]
        tgt_text = item['translation'][config['target_lang']]

        enc_input_tokens = tokenizer_src.encode(src_text).ids
        dec_input_tokens = tokenizer_tgt.encode(tgt_text).ids

        if len(enc_input_tokens) > max_len_src:
            max_len_src = len(enc_input_tokens)
        if len(dec_input_tokens) > max_len_tgt:
            max_len_tgt = len(dec_input_tokens)
    
    print(f"Max source length (in filtered sample): {max_len_src}, Max target length: {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_size_len, vocab_tgt_len):
    model = build_transformer(
        src_vocab_size=vocab_size_len,
        tgt_vocab_size=vocab_tgt_len,
        src_seq_len=config['seq_length'],
        tgt_seq_len=config['seq_length'],
        d_model=config['d_model'],
        num_heads=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['dim_feedforward'],
        dropout=config['dropout']
    )
    return model

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(f"file:{Path.cwd()}/mlruns")
    mlflow.set_experiment(config['experiment_name'])
    
    Path(config['model_folder'] + '/' + config['experiment_name']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, len(tokenizer_src.get_vocab()), len(tokenizer_tgt.get_vocab()))
    model.to(device)    

    writer = SummaryWriter(log_dir=config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1)
    
    initial_epoch = 0
    global_step = 0
    
    if config['preload'] is not None:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Loading model weights from {model_filename}...")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch']
        global_step = state['global_step']
    
    with mlflow.start_run():
        # Log all configuration parameters
        mlflow.log_params(config)
        
        for epoch in range(initial_epoch, config['num_epochs']):
            model.train()
            batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['enc_mask'].to(device)
            decoder_mask = batch['dec_mask'].to(device)   
            
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            projected_output = model.project(decoder_output)

            label = batch['label'].to(device)
            loss = criterion(projected_output.view(-1, projected_output.size(-1)), label.view(-1))

            # Check for NaN
            if torch.isnan(loss):
                print("WARNING: NaN loss detected! Skipping batch...")
                continue

            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            mlflow.log_metric('train_loss', loss.item(), step=global_step)
            writer.flush()

            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Run validation at the end of each epoch
        run_validation(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config['seq_length'],
            device,
            lambda msg: print(msg),
            global_step,
            writer,
            num_examples=2
        )
        
        # Save model checkpoint
        model_filename = get_weights_file_path(config, f"{epoch+1:02d}")
        print(f"Saving model checkpoint to {model_filename}...")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)
        
        # Log model checkpoint and metrics at end of epoch
        mlflow.log_artifact(model_filename, artifact_path="model_checkpoints")
        mlflow.log_metric('epoch', epoch + 1)
        
        # Log model itself to MLflow
        mlflow.pytorch.log_model(model, "model_latest")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
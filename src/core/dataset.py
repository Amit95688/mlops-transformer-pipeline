import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, source_lang, target_lang, seq_length):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.seq_length = seq_length
        
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.long)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.long)
        self.pad_token_id = tokenizer_tgt.token_to_id("[PAD]")

    def __len__(self):
        return len(self.ds)     
    
    def __getitem__(self, idx): 
        item = self.ds[idx]
        src_text = item['translation'][self.source_lang]
        tgt_text = item['translation'][self.target_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        enc_num_padding_tokens = self.seq_length - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_length - len(dec_input_tokens) - 1 
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            # Truncate instead of raising error
            enc_input_tokens = enc_input_tokens[:self.seq_length - 2]
            dec_input_tokens = dec_input_tokens[:self.seq_length - 1]
            enc_num_padding_tokens = self.seq_length - len(enc_input_tokens) - 2
            dec_num_padding_tokens = self.seq_length - len(dec_input_tokens) - 1
        
        enc_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.long),
            self.eos_token,
            torch.tensor([self.pad_token_id] * enc_num_padding_tokens, dtype=torch.long)
        ])
        
        dec_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.long),
            torch.tensor([self.pad_token_id] * dec_num_padding_tokens, dtype=torch.long)
        ])
        
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.long),
            self.eos_token,
            torch.tensor([self.pad_token_id] * dec_num_padding_tokens, dtype=torch.long)
        ])
        
        assert enc_input.size(0) == self.seq_length
        assert dec_input.size(0) == self.seq_length
        assert label.size(0) == self.seq_length
        
        # Create encoder mask: (1, 1, seq_len) - for padding only
        enc_mask = (enc_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int()
        
        # Create decoder mask: (1, seq_len, seq_len) - for padding + causal
        # Step 1: Padding mask (seq_len,)
        dec_padding_mask = (dec_input != self.pad_token_id)
        
        # Step 2: Get causal mask (1, seq_len, seq_len)
        causal = casual_mask(dec_input.size(0))
        
        # Step 3: Expand padding mask to 2D: (seq_len,) -> (1, 1, seq_len) -> broadcast with causal
        # We need padding mask to be shape (1, seq_len, 1) so it broadcasts with (1, seq_len, seq_len)
        dec_padding_mask_expanded = dec_padding_mask.unsqueeze(0).unsqueeze(-1).int()  # (1, seq_len, 1)
        
        # Combine: both padding positions AND causal masking
        dec_mask = causal & dec_padding_mask_expanded  # (1, seq_len, seq_len)
        
        return {
            "encoder_input": enc_input,  # (seq_len)
            "decoder_input": dec_input,  # (seq_len)
            "enc_mask": enc_mask,  # (1, 1, seq_len)
            "dec_mask": dec_mask,  # (1, seq_len, seq_len)
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }   

def casual_mask(size):
    """
    Creates a causal (lower triangular) mask.
    Returns shape: (1, size, size)
    
    Example for size=4:
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
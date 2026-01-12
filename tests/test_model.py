"""Tests for model components"""
import pytest
import torch
from pathlib import Path

from model import build_transformer


class TestModelBuilding:
    """Test model building and architecture"""
    
    def test_build_transformer(self, config):
        """Test building transformer model"""
        model = build_transformer(
            src_vocab_size=5000,
            tgt_vocab_size=5000,
            src_seq_len=512,
            tgt_seq_len=512,
            d_model=config["d_model"],
            d_ff=config["dim_feedforward"],
            num_heads=config["nhead"],
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            dropout=config["dropout"],
        )
        
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_model_forward_pass(self, config, device):
        """Test forward pass through model"""
        model = build_transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            src_seq_len=64,
            tgt_seq_len=64,
            d_model=config["d_model"],
            d_ff=config["dim_feedforward"],
            num_heads=config["nhead"],
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout=0.1,
        ).to(device)
        
        batch_size = 2
        src_ids = torch.randint(0, 1000, (batch_size, 64)).to(device)
        tgt_ids = torch.randint(0, 1000, (batch_size, 64)).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model.encode(src_ids)
        
        assert output.shape[0] == batch_size
        assert output.shape[2] == config["d_model"]
    
    def test_model_has_required_components(self, config):
        """Test model has encoder and decoder"""
        model = build_transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            src_seq_len=64,
            tgt_seq_len=64,
            d_model=config["d_model"],
            d_ff=config["dim_feedforward"],
            num_heads=config["nhead"],
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout=0.1,
        )
        
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')

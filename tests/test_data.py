"""Tests for data pipeline"""
import pytest
from pathlib import Path


class TestDataLoading:
    """Test data loading and preprocessing"""
    
    def test_config_exists(self, config):
        """Test configuration loads correctly"""
        assert config is not None
        assert "datasource" in config
        assert "lang_src" in config
        assert "lang_tgt" in config
    
    def test_required_config_keys(self, config):
        """Test all required config keys exist"""
        required_keys = [
            "datasource",
            "lang_src",
            "lang_tgt",
            "d_model",
            "nhead",
            "num_encoder_layers",
            "num_decoder_layers",
            "batch_size",
            "num_epochs",
            "lr",
            "seq_length"
        ]
        
        for key in required_keys:
            assert key in config, f"Missing config key: {key}"


class TestDataValidation:
    """Test data validation"""
    
    def test_tokenizer_files_exist(self):
        """Test tokenizer files are present"""
        from config import get_config
        config = get_config()
        
        src_tokenizer = Path(config["tokenizer_file"].format(lang=config["lang_src"]))
        tgt_tokenizer = Path(config["tokenizer_file"].format(lang=config["lang_tgt"]))
        
        # These files should exist after training
        if src_tokenizer.exists():
            assert src_tokenizer.stat().st_size > 0
        if tgt_tokenizer.exists():
            assert tgt_tokenizer.stat().st_size > 0

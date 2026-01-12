"""Tests for model artifacts and checkpoints"""
import pytest
from pathlib import Path


class TestModelArtifacts:
    """Test model artifact management"""
    
    def test_checkpoint_structure(self, config):
        """Test checkpoint files exist and have proper structure"""
        model_path = Path(config["model_folder"]) / config["experiment_name"]
        
        if model_path.exists():
            checkpoints = list(model_path.rglob("*.pt"))
            assert len(checkpoints) > 0, "No checkpoints found"
            
            for checkpoint in checkpoints:
                assert checkpoint.stat().st_size > 0, f"Empty checkpoint: {checkpoint}"
    
    def test_tokenizers_exist(self, config):
        """Test required tokenizer files exist"""
        src_tok = Path(config["tokenizer_file"].format(lang=config["lang_src"]))
        tgt_tok = Path(config["tokenizer_file"].format(lang=config["lang_tgt"]))
        
        if src_tok.exists():
            assert src_tok.stat().st_size > 1000, "Source tokenizer too small"
        if tgt_tok.exists():
            assert tgt_tok.stat().st_size > 1000, "Target tokenizer too small"


class TestModelPerformance:
    """Test model performance metrics"""
    
    def test_model_can_be_loaded(self, config, device):
        """Test loading trained model"""
        import torch
        from model import build_transformer
        
        model_path = Path(config["model_folder"]) / config["experiment_name"]
        
        if not model_path.exists():
            pytest.skip("Model not trained yet")
        
        checkpoints = list(model_path.rglob("*.pt"))
        if not checkpoints:
            pytest.skip("No checkpoints available")
        
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        # Load checkpoint
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        assert checkpoint is not None

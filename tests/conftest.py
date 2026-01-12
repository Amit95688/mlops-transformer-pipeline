"""Pytest configuration and fixtures"""
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def config():
    """Load config for testing"""
    from config import get_config
    return get_config()


@pytest.fixture
def device():
    """Get test device"""
    import torch
    return torch.device("cpu")

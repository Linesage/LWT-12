import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hdc import DEV, HDCConfig, HDCBitTransformer  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def small_cfg():
    """Малая конфигурация: H кратно 16 для fp8-пути, задача помещается в память."""
    return HDCConfig(d_hidden=1024, n_layers=2, n_slots=256, topk_slots=7,
                     topk_attn=4, vocab=32, seq_len=16)


@pytest.fixture
def model(small_cfg):
    return HDCBitTransformer(small_cfg, seed=0)


@pytest.fixture
def gen():
    return torch.Generator(device=DEV).manual_seed(1)

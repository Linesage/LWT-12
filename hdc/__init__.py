"""Бинарный HDC-трансформер: обучение точным backward над GF(2)."""

from .config import DEV, SRC_WEIGHT, HDCConfig
from .model import BIT_ROLES, BitParam, BundleSite, HammingTrace, HDCBitTransformer
from .primitives import (bind, bipolar_bmm, bipolar_mm, bundle,
                         hamming, pack_bits, permute, res_weight, shift_next,
                         shift_prev, topk_select, unpack_bits)
from .errors import (apply_flips, consensus_flip_masks, exact_backward,
                     split_by_margin, target_error_mask, wanted_state)

__all__ = [
    "DEV", "HDCConfig", "HDCBitTransformer", "BitParam", "BundleSite",
    "HammingTrace", "BIT_ROLES", "SRC_WEIGHT", "bind", "permute", "bundle",
    "hamming", "bipolar_mm", "bipolar_bmm", "res_weight", "shift_prev",
    "shift_next", "topk_select", "pack_bits", "unpack_bits", "exact_backward",
    "wanted_state", "target_error_mask", "split_by_margin", "apply_flips",
    "consensus_flip_masks",
]

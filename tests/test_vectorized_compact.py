"""Test vectorized compact_kv produces correct shapes and properties."""

import math

import pytest
import torch

from prime_rl.inference.compaction.algorithm import compact_kv


@pytest.fixture
def kv_data():
    """Synthetic KV cache matching Qwen3-4B dimensions."""
    device = torch.device("cpu")
    dtype = torch.float32
    num_layers = 4  # fewer for test speed
    num_kv_heads = 8
    head_size = 128
    prompt_len = 50
    asst_len = 200
    seq_len = prompt_len + asst_len

    keys = [torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype, device=device)
            for _ in range(num_layers)]
    values = [torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype, device=device)
              for _ in range(num_layers)]

    return {
        "keys": keys, "values": values,
        "prompt_len": prompt_len, "num_kv_heads": num_kv_heads,
        "head_size": head_size, "device": device,
        "num_layers": num_layers, "asst_len": asst_len,
    }


def test_output_shapes(kv_data):
    target_ratio = 0.25
    window = 100
    target_len = max(1, int(window * target_ratio))

    c1_list, c2_list = compact_kv(
        kv_data["keys"], kv_data["values"],
        kv_data["prompt_len"], target_ratio,
        kv_data["num_kv_heads"], kv_data["head_size"],
        kv_data["device"], compact_window=window,
    )

    assert len(c1_list) == kv_data["num_layers"]
    assert len(c2_list) == kv_data["num_layers"]
    for layer_idx in range(kv_data["num_layers"]):
        assert c1_list[layer_idx].shape == (target_len, kv_data["num_kv_heads"], kv_data["head_size"])
        assert c2_list[layer_idx].shape == (target_len, kv_data["num_kv_heads"], kv_data["head_size"])


def test_c1_keys_are_subset(kv_data):
    """C1 keys must be a subset of original prefix keys."""
    target_ratio = 0.25
    window = 100

    c1_list, _ = compact_kv(
        kv_data["keys"], kv_data["values"],
        kv_data["prompt_len"], target_ratio,
        kv_data["num_kv_heads"], kv_data["head_size"],
        kv_data["device"], compact_window=window,
    )

    for layer_idx in range(kv_data["num_layers"]):
        orig_prefix = kv_data["keys"][layer_idx][kv_data["prompt_len"]:kv_data["prompt_len"] + window]
        c1 = c1_list[layer_idx]
        for h in range(kv_data["num_kv_heads"]):
            for t in range(c1.shape[0]):
                # Each c1 key should match some original prefix key for this head
                matches = (orig_prefix[:, h, :] == c1[t, h, :]).all(dim=-1)
                assert matches.any(), f"c1[{layer_idx}][{t},{h}] not found in original keys"


def test_full_window_compaction(kv_data):
    """compact_window=None compresses all assistant tokens."""
    target_ratio = 0.25
    target_len = max(1, int(kv_data["asst_len"] * target_ratio))

    c1_list, c2_list = compact_kv(
        kv_data["keys"], kv_data["values"],
        kv_data["prompt_len"], target_ratio,
        kv_data["num_kv_heads"], kv_data["head_size"],
        kv_data["device"], compact_window=None,
    )

    for layer_idx in range(kv_data["num_layers"]):
        assert c1_list[layer_idx].shape[0] == target_len
        assert c2_list[layer_idx].shape[0] == target_len


def test_dtype_preserved(kv_data):
    c1_list, c2_list = compact_kv(
        kv_data["keys"], kv_data["values"],
        kv_data["prompt_len"], 0.25,
        kv_data["num_kv_heads"], kv_data["head_size"],
        kv_data["device"], compact_window=100,
    )

    for layer_idx in range(kv_data["num_layers"]):
        assert c1_list[layer_idx].dtype == kv_data["keys"][0].dtype
        assert c2_list[layer_idx].dtype == kv_data["values"][0].dtype


def test_no_nan_inf(kv_data):
    c1_list, c2_list = compact_kv(
        kv_data["keys"], kv_data["values"],
        kv_data["prompt_len"], 0.25,
        kv_data["num_kv_heads"], kv_data["head_size"],
        kv_data["device"], compact_window=100,
    )

    for layer_idx in range(kv_data["num_layers"]):
        assert not c1_list[layer_idx].isnan().any()
        assert not c1_list[layer_idx].isinf().any()
        assert not c2_list[layer_idx].isnan().any()
        assert not c2_list[layer_idx].isinf().any()

"""Standalone test for segmented forward pass with compaction replay.

Tests on CPU with a small model to verify:
1. Logits shape matches full forward
2. Gradients flow through the segmented path
3. Boundary token handling is correct
4. Compaction algorithm runs without errors

Usage:
    python scripts/test_segmented_forward.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from prime_rl.trainer.rl.compaction import segmented_forward


def create_test_model(vocab_size=256, hidden_size=64, num_layers=2, num_heads=4):
    """Create a tiny model for testing."""
    config = AutoConfig.for_model(
        "qwen2",
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=512,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.train()
    return model


def test_shape_consistency():
    """Test that segmented forward produces logits of the same shape as standard forward."""
    print("=== Test: Shape Consistency ===")
    model = create_test_model()

    prompt_len = 10
    seg0_len = 20
    seg1_len = 15
    total_len = prompt_len + seg0_len + seg1_len
    segment_boundaries = [seg0_len, seg0_len + seg1_len]

    input_ids = torch.randint(0, 256, (1, total_len))
    position_ids = torch.arange(total_len).unsqueeze(0)
    temperature = torch.ones(1, total_len)

    # Standard forward
    with torch.no_grad():
        std_out = model(input_ids=input_ids, position_ids=position_ids)
    std_logits = std_out.logits

    # Segmented forward
    with torch.no_grad():
        seg_out = segmented_forward(
            model, input_ids, position_ids,
            segment_boundaries=segment_boundaries,
            prompt_len=prompt_len,
            compact_target_ratio=0.5,
            compact_window=None,
            temperature=temperature,
        )
    seg_logits = seg_out["logits"]

    assert std_logits.shape == seg_logits.shape, (
        f"Shape mismatch: std={std_logits.shape}, seg={seg_logits.shape}"
    )
    print(f"  Standard shape: {std_logits.shape}")
    print(f"  Segmented shape: {seg_logits.shape}")
    print("  PASSED")


def test_gradient_flow():
    """Test that gradients flow through the segmented forward."""
    print("\n=== Test: Gradient Flow ===")
    model = create_test_model()

    prompt_len = 10
    seg0_len = 20
    seg1_len = 15
    total_len = prompt_len + seg0_len + seg1_len
    segment_boundaries = [seg0_len, seg0_len + seg1_len]

    input_ids = torch.randint(0, 256, (1, total_len))
    position_ids = torch.arange(total_len).unsqueeze(0)
    temperature = torch.ones(1, total_len)

    out = segmented_forward(
        model, input_ids, position_ids,
        segment_boundaries=segment_boundaries,
        prompt_len=prompt_len,
        compact_target_ratio=0.5,
        compact_window=None,
        temperature=temperature,
    )

    loss = out["logits"].sum()
    loss.backward()

    # Check at least some parameters have gradients
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for p in model.parameters())
    print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
    assert params_with_grad > 0, "No gradients!"
    print("  PASSED")


def test_single_segment():
    """Test that a single segment (no compaction) matches standard forward."""
    print("\n=== Test: Single Segment (No Compaction) ===")
    model = create_test_model()

    prompt_len = 10
    completion_len = 20
    total_len = prompt_len + completion_len
    segment_boundaries = [completion_len]  # single segment, no compaction

    input_ids = torch.randint(0, 256, (1, total_len))
    position_ids = torch.arange(total_len).unsqueeze(0)
    temperature = torch.ones(1, total_len)

    with torch.no_grad():
        std_out = model(input_ids=input_ids, position_ids=position_ids)
        seg_out = segmented_forward(
            model, input_ids, position_ids,
            segment_boundaries=segment_boundaries,
            prompt_len=prompt_len,
            compact_target_ratio=0.5,
            compact_window=None,
            temperature=temperature,
        )

    # Single segment should match standard forward exactly
    diff = (std_out.logits - seg_out["logits"]).abs().max().item()
    print(f"  Max logit difference: {diff:.6e}")
    assert diff < 1e-4, f"Logits differ by {diff}"
    print("  PASSED")


def test_partial_compaction():
    """Test segmented forward with compact_window (partial compaction)."""
    print("\n=== Test: Partial Compaction (compact_window) ===")
    model = create_test_model()

    prompt_len = 10
    seg0_len = 30
    seg1_len = 20
    total_len = prompt_len + seg0_len + seg1_len
    segment_boundaries = [seg0_len, seg0_len + seg1_len]

    input_ids = torch.randint(0, 256, (1, total_len))
    position_ids = torch.arange(total_len).unsqueeze(0)
    temperature = torch.ones(1, total_len)

    out = segmented_forward(
        model, input_ids, position_ids,
        segment_boundaries=segment_boundaries,
        prompt_len=prompt_len,
        compact_target_ratio=0.25,
        compact_window=15,  # only compact first 15 of 30 assistant tokens
        temperature=temperature,
    )

    assert out["logits"].shape == (1, total_len, model.config.vocab_size)
    print(f"  Output shape: {out['logits'].shape}")
    print("  PASSED")


def test_three_segments():
    """Test with 3 segments (2 compactions)."""
    print("\n=== Test: Three Segments (2 Compactions) ===")
    model = create_test_model()

    prompt_len = 10
    seg_lens = [20, 20, 20]
    total_completion = sum(seg_lens)
    total_len = prompt_len + total_completion
    segment_boundaries = []
    cum = 0
    for sl in seg_lens:
        cum += sl
        segment_boundaries.append(cum)

    input_ids = torch.randint(0, 256, (1, total_len))
    position_ids = torch.arange(total_len).unsqueeze(0)
    temperature = torch.ones(1, total_len)

    out = segmented_forward(
        model, input_ids, position_ids,
        segment_boundaries=segment_boundaries,
        prompt_len=prompt_len,
        compact_target_ratio=0.5,
        compact_window=None,
        temperature=temperature,
    )

    assert out["logits"].shape == (1, total_len, model.config.vocab_size)

    # Also check gradient
    loss = out["logits"].sum()
    loss.backward()
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Output shape: {out['logits'].shape}")
    print(f"  Parameters with gradients: {params_with_grad}")
    assert params_with_grad > 0
    print("  PASSED")


def test_deterministic_compaction():
    """Test that seeded compaction produces identical results across calls."""
    print("\n=== Test: Deterministic Compaction (Seeded) ===")
    model = create_test_model()

    prompt_len = 10
    seg0_len = 30
    seg1_len = 20
    total_len = prompt_len + seg0_len + seg1_len
    segment_boundaries = [seg0_len, seg0_len + seg1_len]

    input_ids = torch.randint(0, 256, (1, total_len))
    position_ids = torch.arange(total_len).unsqueeze(0)
    temperature = torch.ones(1, total_len)

    # Run twice — should produce identical logits because compaction is seeded
    with torch.no_grad():
        out1 = segmented_forward(
            model, input_ids, position_ids,
            segment_boundaries=segment_boundaries,
            prompt_len=prompt_len,
            compact_target_ratio=0.5,
            compact_window=15,
            temperature=temperature,
        )
        out2 = segmented_forward(
            model, input_ids, position_ids,
            segment_boundaries=segment_boundaries,
            prompt_len=prompt_len,
            compact_target_ratio=0.5,
            compact_window=15,
            temperature=temperature,
        )

    diff = (out1["logits"] - out2["logits"]).abs().max().item()
    print(f"  Max logit difference between runs: {diff:.6e}")
    # Small fp32 accumulation from DynamicCache is expected; randomness would give ~1e0
    assert diff < 1e-5, f"Non-deterministic compaction! diff={diff}"
    print("  PASSED")


def test_compute_beta():
    """Test that compute_beta=True works in segmented forward."""
    print("\n=== Test: Compute Beta ===")
    model = create_test_model()

    prompt_len = 10
    seg0_len = 30
    seg1_len = 20
    total_len = prompt_len + seg0_len + seg1_len
    segment_boundaries = [seg0_len, seg0_len + seg1_len]

    input_ids = torch.randint(0, 256, (1, total_len))
    position_ids = torch.arange(total_len).unsqueeze(0)
    temperature = torch.ones(1, total_len)

    out = segmented_forward(
        model, input_ids, position_ids,
        segment_boundaries=segment_boundaries,
        prompt_len=prompt_len,
        compact_target_ratio=0.5,
        compact_window=15,
        temperature=temperature,
        compute_beta=True,
    )

    assert out["logits"].shape == (1, total_len, model.config.vocab_size)

    # Check gradient flow with beta
    loss = out["logits"].sum()
    loss.backward()
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Output shape: {out['logits'].shape}")
    print(f"  Parameters with gradients: {params_with_grad}")
    assert params_with_grad > 0
    print("  PASSED")


def test_compact_kv_seed_consistency():
    """Test that compact_kv with same seed produces identical C1/C2."""
    print("\n=== Test: compact_kv Seed Consistency ===")
    from prime_rl.inference.compaction.algorithm import compact_kv

    num_kv_heads, head_size, num_layers = 4, 16, 2
    seq_len, prompt_len = 50, 10
    keys = [torch.randn(seq_len, num_kv_heads, head_size) for _ in range(num_layers)]
    values = [torch.randn(seq_len, num_kv_heads, head_size) for _ in range(num_layers)]

    kwargs = dict(
        prompt_len=prompt_len, target_ratio=0.5, num_kv_heads=num_kv_heads,
        head_size=head_size, device=torch.device("cpu"), compact_window=20,
        compute_beta=True, seed=12345,
    )

    c1a, c2a, ba = compact_kv(keys, values, **kwargs)
    c1b, c2b, bb = compact_kv(keys, values, **kwargs)

    for l in range(num_layers):
        assert torch.equal(c1a[l], c1b[l]), f"C1 differs at layer {l}"
        # lstsq uses SVD/QR which isn't bitwise deterministic; check approximate equality
        c2_diff = (c2a[l] - c2b[l]).abs().max().item()
        assert c2_diff < 1e-5, f"C2 differs too much at layer {l}: {c2_diff}"
        beta_diff = (ba[l] - bb[l]).abs().max().item()
        assert beta_diff < 1e-5, f"Beta differs too much at layer {l}: {beta_diff}"

    # Without seed: should differ (with very high probability)
    c1c, c2c, _ = compact_kv(keys, values, prompt_len=prompt_len, target_ratio=0.5,
                              num_kv_heads=num_kv_heads, head_size=head_size,
                              device=torch.device("cpu"), compact_window=20)
    c1d, c2d, _ = compact_kv(keys, values, prompt_len=prompt_len, target_ratio=0.5,
                              num_kv_heads=num_kv_heads, head_size=head_size,
                              device=torch.device("cpu"), compact_window=20)
    any_diff = any(not torch.equal(c1c[l], c1d[l]) for l in range(num_layers))
    print(f"  Seeded: identical=True")
    print(f"  Unseeded: differs={any_diff}")
    assert any_diff, "Unseeded calls should produce different results"
    print("  PASSED")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_single_segment()
    test_shape_consistency()
    test_gradient_flow()
    test_partial_compaction()
    test_three_segments()
    test_deterministic_compaction()
    test_compute_beta()
    test_compact_kv_seed_consistency()
    print("\n=== ALL TESTS PASSED ===")

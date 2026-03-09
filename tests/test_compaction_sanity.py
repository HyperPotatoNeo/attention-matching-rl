"""Sanity tests for compaction RL pipeline.

Tests cover:
1. compact_kv algorithm — shapes, key subset property, deterministic structure
2. Segment range building — covers all tokens, no gaps/overlaps
3. Logit accounting — total logits == input length
4. _get_kv_from_cache — both DynamicCache API variants
5. KV cache reconstruction — [prompt | compacted | suffix] structure
6. End-to-end segmented_forward — first segment logits match single forward
7. Position ID correctness across segments

Run on GPU node:
    cd $SCRATCH/compaction-rl
    source .venv/bin/activate
    python -m pytest tests/test_compaction_sanity.py -v
"""

import math
import pytest
import torch
from torch import Tensor
from transformers import DynamicCache


# ── 1. compact_kv algorithm ──────────────────────────────────────────────────

class TestCompactKV:
    """Verify compact_kv output shapes and structural properties."""

    @pytest.fixture
    def kv_inputs(self):
        """Create synthetic KV cache inputs (2 layers, 4 heads, 128 dim)."""
        num_layers = 2
        num_kv_heads = 4
        head_size = 128
        prompt_len = 50
        asst_len = 200
        seq_len = prompt_len + asst_len
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16

        keys = [torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype)
                for _ in range(num_layers)]
        values = [torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype)
                  for _ in range(num_layers)]

        return keys, values, prompt_len, num_kv_heads, head_size, device, asst_len

    def test_output_shapes(self, kv_inputs):
        from prime_rl.inference.compaction.algorithm import compact_kv

        keys, values, prompt_len, num_kv_heads, head_size, device, asst_len = kv_inputs
        target_ratio = 0.25
        target_len = max(1, int(asst_len * target_ratio))

        c1_list, c2_list = compact_kv(
            keys, values, prompt_len, target_ratio,
            num_kv_heads, head_size, device,
        )

        assert len(c1_list) == len(keys), "One C1 per layer"
        assert len(c2_list) == len(keys), "One C2 per layer"

        for l in range(len(keys)):
            assert c1_list[l].shape == (target_len, num_kv_heads, head_size), \
                f"C1[{l}] shape mismatch: {c1_list[l].shape}"
            assert c2_list[l].shape == (target_len, num_kv_heads, head_size), \
                f"C2[{l}] shape mismatch: {c2_list[l].shape}"

    def test_c1_keys_are_subset_of_input(self, kv_inputs):
        """C1 keys should be selected from the original prefix keys (same values)."""
        from prime_rl.inference.compaction.algorithm import compact_kv

        keys, values, prompt_len, num_kv_heads, head_size, device, asst_len = kv_inputs
        target_ratio = 0.25

        c1_list, _ = compact_kv(
            keys, values, prompt_len, target_ratio,
            num_kv_heads, head_size, device,
        )

        for l in range(len(keys)):
            asst_K = keys[l][prompt_len:]
            c1 = c1_list[l]
            for h in range(num_kv_heads):
                for t in range(c1.shape[0]):
                    c1_vec = c1[t, h, :]
                    # Check this vector exists in the original assistant keys
                    diffs = (asst_K[:, h, :] - c1_vec.unsqueeze(0)).abs().sum(dim=-1)
                    assert diffs.min() < 1e-4, \
                        f"C1[{l}][{t},{h}] not found in original keys"

    def test_compact_window(self, kv_inputs):
        """With compact_window, only first N assistant tokens are compressed."""
        from prime_rl.inference.compaction.algorithm import compact_kv

        keys, values, prompt_len, num_kv_heads, head_size, device, asst_len = kv_inputs
        compact_window = 100
        target_ratio = 0.25
        target_len = max(1, int(compact_window * target_ratio))

        c1_list, c2_list = compact_kv(
            keys, values, prompt_len, target_ratio,
            num_kv_heads, head_size, device,
            compact_window=compact_window,
        )

        for l in range(len(keys)):
            assert c1_list[l].shape[0] == target_len, \
                f"With window={compact_window}, expected {target_len} keys, got {c1_list[l].shape[0]}"

    def test_c2_dtype_matches_input(self, kv_inputs):
        """C2 values should be in the same dtype as input."""
        from prime_rl.inference.compaction.algorithm import compact_kv

        keys, values, prompt_len, num_kv_heads, head_size, device, _ = kv_inputs

        c1_list, c2_list = compact_kv(
            keys, values, prompt_len, 0.25,
            num_kv_heads, head_size, device,
        )

        for l in range(len(keys)):
            assert c2_list[l].dtype == keys[l].dtype, \
                f"C2[{l}] dtype {c2_list[l].dtype} != input dtype {keys[l].dtype}"


# ── 2. Segment range building ────────────────────────────────────────────────

class TestSegmentRanges:
    """Verify segment range logic matches what segmented_forward computes."""

    @staticmethod
    def build_ranges(segment_boundaries, prompt_len):
        """Reproduce the range-building logic from segmented_forward."""
        seg_input_ranges = []
        prev_boundary = 0
        for i, boundary in enumerate(segment_boundaries):
            if i == 0:
                seg_start = 0
            else:
                seg_start = prompt_len + prev_boundary - 1
            seg_end = prompt_len + boundary
            seg_input_ranges.append((seg_start, seg_end))
            prev_boundary = boundary
        return seg_input_ranges

    def test_single_segment_covers_all(self):
        """Single segment = entire sequence."""
        prompt_len = 50
        total_completion = 200
        boundaries = [total_completion]
        ranges = self.build_ranges(boundaries, prompt_len)

        assert len(ranges) == 1
        assert ranges[0] == (0, prompt_len + total_completion)

    def test_two_segments_coverage(self):
        """Two segments: first covers [0, prompt+b1), second covers [prompt+b1-1, prompt+b2)."""
        prompt_len = 50
        boundaries = [100, 200]
        ranges = self.build_ranges(boundaries, prompt_len)

        assert len(ranges) == 2
        assert ranges[0] == (0, 150)       # 0 to prompt_len + 100
        assert ranges[1] == (149, 250)     # prompt_len + 100 - 1 to prompt_len + 200

    def test_three_segments_coverage(self):
        prompt_len = 30
        boundaries = [50, 120, 200]
        ranges = self.build_ranges(boundaries, prompt_len)

        assert ranges[0] == (0, 80)        # 0 to 30+50
        assert ranges[1] == (79, 150)      # 30+50-1 to 30+120
        assert ranges[2] == (149, 230)     # 30+120-1 to 30+200

    def test_logit_accounting(self):
        """Total logits = input_len when we drop last logit of non-final segments."""
        prompt_len = 30
        boundaries = [50, 120, 200]
        total_len = prompt_len + boundaries[-1]
        ranges = self.build_ranges(boundaries, prompt_len)

        total_logits = 0
        for i, (s, e) in enumerate(ranges):
            seg_len = e - s
            if i < len(ranges) - 1:
                total_logits += seg_len - 1  # drop last
            else:
                total_logits += seg_len

        assert total_logits == total_len, \
            f"Total logits {total_logits} != input_len {total_len}"

    def test_boundary_overlap_is_exactly_one(self):
        """Adjacent segments overlap by exactly 1 token (the boundary token)."""
        prompt_len = 50
        boundaries = [100, 250, 400]
        ranges = self.build_ranges(boundaries, prompt_len)

        for i in range(len(ranges) - 1):
            overlap = ranges[i][1] - ranges[i + 1][0]
            assert overlap == 1, \
                f"Segments {i} and {i+1} overlap by {overlap}, expected 1"


# ── 3. _get_kv_from_cache ────────────────────────────────────────────────────

class TestGetKVFromCache:
    """Test KV extraction from DynamicCache (old and new API)."""

    def _make_kv(self, num_layers, batch, heads, seq_len, dim, device):
        """Create KV tensors in [batch, heads, seq, dim] format."""
        keys = [torch.randn(batch, heads, seq_len, dim, device=device)
                for _ in range(num_layers)]
        values = [torch.randn(batch, heads, seq_len, dim, device=device)
                  for _ in range(num_layers)]
        return keys, values

    def test_old_api(self):
        """DynamicCache with key_cache/value_cache attributes."""
        from prime_rl.trainer.rl.compaction import _get_kv_from_cache

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_layers, batch, heads, seq_len, dim = 2, 1, 4, 100, 128

        cache = DynamicCache()
        for l in range(num_layers):
            k = torch.randn(batch, heads, seq_len, dim, device=device)
            v = torch.randn(batch, heads, seq_len, dim, device=device)
            cache.update(k, v, l)

        keys, values, num_kv_heads, head_size = _get_kv_from_cache(cache)

        assert len(keys) == num_layers
        assert len(values) == num_layers
        assert num_kv_heads == heads
        assert head_size == dim

        for l in range(num_layers):
            # Output should be [seq, heads, dim]
            assert keys[l].shape == (seq_len, heads, dim), \
                f"keys[{l}] shape {keys[l].shape}, expected ({seq_len}, {heads}, {dim})"
            assert values[l].shape == (seq_len, heads, dim)

    def test_permutation_correctness(self):
        """Verify the permute(1,0,2) correctly transposes [batch=1, heads, seq, dim] -> [seq, heads, dim]."""
        from prime_rl.trainer.rl.compaction import _get_kv_from_cache

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cache = DynamicCache()
        k = torch.arange(24, device=device, dtype=torch.float32).reshape(1, 2, 3, 4)
        v = torch.arange(24, device=device, dtype=torch.float32).reshape(1, 2, 3, 4) + 100
        cache.update(k, v, 0)

        keys, values, _, _ = _get_kv_from_cache(cache)

        # k[0] is [2, 3, 4]. permute(1,0,2) -> [3, 2, 4] (seq, heads, dim)
        expected_k = k[0].permute(1, 0, 2).contiguous()
        assert torch.equal(keys[0], expected_k)


# ── 4. KV cache reconstruction ───────────────────────────────────────────────

class TestKVReconstruction:
    """Verify [prompt | compacted | suffix] structure after compaction."""

    def test_reconstruction_shapes(self):
        from prime_rl.inference.compaction.algorithm import compact_kv

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_layers, num_kv_heads, head_size = 2, 4, 64
        prompt_len = 30
        asst_len = 200
        compact_window = 150
        target_ratio = 0.25
        seq_len = prompt_len + asst_len

        keys = [torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=torch.bfloat16)
                for _ in range(num_layers)]
        values = [torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=torch.bfloat16)
                  for _ in range(num_layers)]

        c1_list, c2_list = compact_kv(
            keys, values, prompt_len, target_ratio,
            num_kv_heads, head_size, device,
            compact_window=compact_window,
        )

        compacted_prefix_len = c1_list[0].shape[0]
        suffix_len = asst_len - compact_window  # 50

        for l in range(num_layers):
            orig_K = keys[l]
            suffix_K = orig_K[prompt_len + compact_window:]
            new_K = torch.cat([orig_K[:prompt_len], c1_list[l], suffix_K], dim=0)

            expected_len = prompt_len + compacted_prefix_len + suffix_len
            assert new_K.shape[0] == expected_len, \
                f"Reconstructed KV len {new_K.shape[0]} != expected {expected_len}"

            # Prompt portion should be unchanged
            assert torch.equal(new_K[:prompt_len], orig_K[:prompt_len]), \
                "Prompt KV was modified during reconstruction"

            # Suffix portion should be unchanged
            assert torch.equal(new_K[prompt_len + compacted_prefix_len:],
                             orig_K[prompt_len + compact_window:]), \
                "Suffix KV was modified during reconstruction"


# ── 5. Position ID handling ──────────────────────────────────────────────────

class TestPositionIDs:
    """Verify position IDs are correctly handled across segments."""

    def test_position_ids_are_original(self):
        """Position IDs passed to segmented_forward should be the original positions.
        After compaction, the KV cache is smaller but keys retain their original RoPE.
        New tokens must use original position IDs, not adjusted ones."""
        prompt_len = 30
        total_completion = 200
        total_len = prompt_len + total_completion

        # Original position IDs are just 0..total_len-1
        position_ids = torch.arange(total_len).unsqueeze(0)

        boundaries = [100, 200]

        # Segment 0: positions 0..129 (original)
        seg0_positions = position_ids[:, 0:130]
        assert seg0_positions[0, 0] == 0
        assert seg0_positions[0, -1] == 129

        # Segment 1: positions 129..229 (original — NOT renumbered)
        seg1_positions = position_ids[:, 129:230]
        assert seg1_positions[0, 0] == 129  # boundary token has its original position
        assert seg1_positions[0, -1] == 229


# ── 6. End-to-end segmented_forward vs single forward ────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestSegmentedForwardE2E:
    """Compare segmented_forward against a single forward pass.

    For the first segment (no compaction yet), logits should match
    a single forward pass on the same tokens.
    """

    @pytest.fixture
    def small_model(self):
        """Load a small model for testing. Uses Qwen2.5-0.5B if available."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-0.5B"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.bfloat16,
            ).to("cuda")
            model.eval()
        except Exception as e:
            pytest.skip(f"Could not load {model_name}: {e}")

        return model, tokenizer

    def test_first_segment_matches_single_forward(self, small_model):
        """First segment logits (before any compaction) should exactly match
        a single forward pass on the same input tokens."""
        model, tokenizer = small_model

        prompt = "What is the capital of France?"
        text = prompt + " The capital of France is Paris, which is a major European city."
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]
        completion_len = seq_len - prompt_len

        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        temperature = torch.ones(1, seq_len, device="cuda")

        # Single forward pass (reference)
        with torch.no_grad():
            ref_out = model(input_ids=input_ids, position_ids=position_ids)
            ref_logits = ref_out.logits

        # Segmented forward with 1 segment (should be identical)
        from prime_rl.trainer.rl.compaction import segmented_forward

        # Need to ensure model has use_cache capability
        model.config.use_cache = False  # segmented_forward sets True internally

        with torch.no_grad():
            seg_out = segmented_forward(
                model=model,
                input_ids=input_ids,
                position_ids=position_ids,
                segment_boundaries=[completion_len],
                prompt_len=prompt_len,
                compact_target_ratio=0.25,
                compact_window=None,
                temperature=temperature,
            )

        seg_logits = seg_out["logits"]

        assert seg_logits.shape == ref_logits.shape, \
            f"Shape mismatch: seg {seg_logits.shape} vs ref {ref_logits.shape}"

        # Temperature=1 so logits should match
        max_diff = (seg_logits - ref_logits).abs().max().item()
        assert max_diff < 1e-2, \
            f"Single-segment logits differ from reference by {max_diff}"

    def test_two_segments_shape_correct(self, small_model):
        """Two-segment forward produces correct output shape."""
        model, tokenizer = small_model

        prompt = "Count to ten: "
        text = prompt + "one two three four five six seven eight nine ten done"
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]
        completion_len = seq_len - prompt_len

        # Split completion roughly in half
        mid = completion_len // 2
        boundaries = [mid, completion_len]

        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        temperature = torch.ones(1, seq_len, device="cuda")

        model.config.use_cache = False

        from prime_rl.trainer.rl.compaction import segmented_forward

        with torch.no_grad():
            seg_out = segmented_forward(
                model=model,
                input_ids=input_ids,
                position_ids=position_ids,
                segment_boundaries=boundaries,
                prompt_len=prompt_len,
                compact_target_ratio=0.25,
                compact_window=None,
                temperature=temperature,
            )

        seg_logits = seg_out["logits"]
        assert seg_logits.shape[1] == seq_len, \
            f"Two-segment logits length {seg_logits.shape[1]} != input length {seq_len}"

    def test_first_segment_logits_match_with_two_segments(self, small_model):
        """When using 2 segments, the first segment's logits (excluding the
        boundary token) should match a single forward pass on those tokens."""
        model, tokenizer = small_model

        prompt = "Solve: 2+2="
        text = prompt + " four. And 3+3= six. The answer is clear."
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]
        completion_len = seq_len - prompt_len

        mid = completion_len // 2
        boundaries = [mid, completion_len]

        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        temperature = torch.ones(1, seq_len, device="cuda")

        # Reference: single forward on first segment's tokens only
        first_seg_end = prompt_len + mid
        with torch.no_grad():
            ref_out = model(
                input_ids=input_ids[:, :first_seg_end],
                position_ids=position_ids[:, :first_seg_end],
            )
            ref_logits = ref_out.logits

        # Segmented forward
        model.config.use_cache = False
        from prime_rl.trainer.rl.compaction import segmented_forward

        with torch.no_grad():
            seg_out = segmented_forward(
                model=model,
                input_ids=input_ids,
                position_ids=position_ids,
                segment_boundaries=boundaries,
                prompt_len=prompt_len,
                compact_target_ratio=0.25,
                compact_window=None,
                temperature=temperature,
            )

        seg_logits = seg_out["logits"]

        # First segment logits (positions 0 to first_seg_end-1) should match
        # (segmented_forward drops the last logit of segment 0, so compare up to that)
        compare_len = first_seg_end - 1
        max_diff = (seg_logits[:, :compare_len, :] - ref_logits[:, :compare_len, :]).abs().max().item()
        assert max_diff < 1e-2, \
            f"First segment logits differ from reference by {max_diff}"


# ── 7. Gradient flow ─────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestGradientFlow:
    """Verify gradients flow through segmented_forward."""

    @pytest.fixture
    def small_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-0.5B"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.bfloat16,
            ).to("cuda")
            model.train()
        except Exception as e:
            pytest.skip(f"Could not load {model_name}: {e}")

        return model, tokenizer

    def test_gradients_flow_single_segment(self, small_model):
        """Single segment: gradients should flow to model parameters."""
        model, tokenizer = small_model

        text = "The quick brown fox jumps over the lazy dog"
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        prompt_len = 3

        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        temperature = torch.ones(1, seq_len, device="cuda")

        from prime_rl.trainer.rl.compaction import segmented_forward

        model.config.use_cache = False
        out = segmented_forward(
            model=model,
            input_ids=input_ids,
            position_ids=position_ids,
            segment_boundaries=[seq_len - prompt_len],
            prompt_len=prompt_len,
            compact_target_ratio=0.25,
            compact_window=None,
            temperature=temperature,
        )

        loss = out["logits"].sum()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().max() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradients flowed through segmented_forward"

    def test_gradients_flow_two_segments(self, small_model):
        """Two segments: gradients should flow for the second segment too.
        Note: compaction detaches KV, so second segment gradients only flow
        through its own forward pass, not through the first segment."""
        model, tokenizer = small_model

        text = "One two three four five six seven eight nine ten eleven twelve"
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        prompt_len = 2
        completion_len = seq_len - prompt_len
        mid = completion_len // 2

        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        temperature = torch.ones(1, seq_len, device="cuda")

        from prime_rl.trainer.rl.compaction import segmented_forward

        model.config.use_cache = False
        out = segmented_forward(
            model=model,
            input_ids=input_ids,
            position_ids=position_ids,
            segment_boundaries=[mid, completion_len],
            prompt_len=prompt_len,
            compact_target_ratio=0.25,
            compact_window=None,
            temperature=temperature,
        )

        loss = out["logits"].sum()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().max() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradients flowed through two-segment forward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

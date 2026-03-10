"""Tests for beta attention correction during training (forward_pre_hooks).

Tests cover:
1. _BetaTrainingState — shape, GQA expansion, clamping
2. _find_attention_modules — generic detection of q/k/v_proj
3. Beta hook — attention_mask modification with broadcasting
4. segmented_forward with compute_beta=True — logits differ from without beta
5. Gradient flow through beta-corrected segments
6. Config auto-sync from env args to trainer

Run on GPU node:
    cd $SCRATCH/compaction-rl
    source .venv/bin/activate
    python -m pytest tests/test_beta_training.py -v
"""

import pytest
import torch


# ── 1. _BetaTrainingState ───────────────────────────────────────────────────

class TestBetaTrainingState:
    def test_set_beta_shapes(self):
        from prime_rl.trainer.rl.compaction import _BetaTrainingState

        num_kv_heads = 8
        num_heads = 32
        state = _BetaTrainingState(num_kv_heads, num_heads)

        # Simulate compact_kv output: (target_len, num_kv_heads)
        target_len = 25
        total_kv_len = 130
        prompt_len = 50
        beta_list = [torch.randn(target_len, num_kv_heads) for _ in range(4)]

        state.set_beta(beta_list, prompt_len, target_len, total_kv_len, torch.device("cpu"))

        assert state.active
        assert len(state.beta_per_layer) == 4

        for layer_idx in range(4):
            bias = state.beta_per_layer[layer_idx]
            assert bias.shape == (1, num_heads, 1, total_kv_len)
            assert bias.dtype == torch.float32

            # Prompt region should be zero
            assert (bias[0, :, 0, :prompt_len] == 0).all()
            # Suffix region should be zero
            assert (bias[0, :, 0, prompt_len + target_len:] == 0).all()
            # Compacted region should be non-zero (with overwhelming probability)
            assert bias[0, :, 0, prompt_len:prompt_len + target_len].abs().sum() > 0

    def test_gqa_expansion(self):
        """Beta values should be repeated across GQA groups."""
        from prime_rl.trainer.rl.compaction import _BetaTrainingState

        num_kv_heads = 4
        num_heads = 16  # 4 groups of 4
        state = _BetaTrainingState(num_kv_heads, num_heads)

        target_len = 10
        # Each kv-head has a unique constant beta
        beta = torch.zeros(target_len, num_kv_heads)
        for h in range(num_kv_heads):
            beta[:, h] = h + 1.0

        state.set_beta([beta], prompt_len=5, compacted_len=target_len,
                       total_kv_len=20, device=torch.device("cpu"))

        bias = state.beta_per_layer[0]
        # Heads 0-3 should have beta=1.0, heads 4-7 should have beta=2.0, etc.
        for h in range(num_heads):
            kv_head = h // (num_heads // num_kv_heads)
            expected = kv_head + 1.0
            actual = bias[0, h, 0, 5:15]
            assert torch.allclose(actual, torch.full_like(actual, expected)), \
                f"Head {h} (kv_head {kv_head}): expected {expected}, got {actual[0].item()}"

    def test_clamping(self):
        """Beta values should be clamped to [-30, 30]."""
        from prime_rl.trainer.rl.compaction import _BetaTrainingState

        state = _BetaTrainingState(num_kv_heads=2, num_heads=2)
        beta = torch.tensor([[100.0, -100.0]], dtype=torch.float32)  # (1, 2)
        state.set_beta([beta], prompt_len=0, compacted_len=1,
                       total_kv_len=5, device=torch.device("cpu"))

        bias = state.beta_per_layer[0]
        assert bias[0, 0, 0, 0].item() == pytest.approx(30.0)
        assert bias[0, 1, 0, 0].item() == pytest.approx(-30.0)

    def test_clear(self):
        from prime_rl.trainer.rl.compaction import _BetaTrainingState

        state = _BetaTrainingState(num_kv_heads=2, num_heads=4)
        state.set_beta([torch.randn(5, 2)], 0, 5, 10, torch.device("cpu"))
        assert state.active

        state.clear()
        assert not state.active
        assert len(state.beta_per_layer) == 0


# ── 2. _find_attention_modules ──────────────────────────────────────────────

class TestFindAttentionModules:
    def test_finds_modules_with_qkv_proj(self):
        from prime_rl.trainer.rl.compaction import _find_attention_modules

        class FakeAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(4, 4)
                self.k_proj = torch.nn.Linear(4, 4)
                self.v_proj = torch.nn.Linear(4, 4)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn0 = FakeAttn()
                self.attn1 = FakeAttn()
                self.mlp = torch.nn.Linear(4, 4)

        model = FakeModel()
        modules = _find_attention_modules(model)
        assert len(modules) == 2
        assert modules[0] is model.attn0
        assert modules[1] is model.attn1

    def test_ignores_modules_without_qkv(self):
        from prime_rl.trainer.rl.compaction import _find_attention_modules

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.norm = torch.nn.LayerNorm(4)

        model = FakeModel()
        assert _find_attention_modules(model) == []


# ── 3. Beta hook logic ─────────────────────────────────────────────────────

class TestBetaHook:
    def test_hook_modifies_attention_mask(self):
        from prime_rl.trainer.rl.compaction import _BetaTrainingState, _make_beta_hook

        num_kv_heads = 2
        num_heads = 4
        state = _BetaTrainingState(num_kv_heads, num_heads)

        # Set a known beta: all 1.0 at compacted positions
        beta = torch.ones(3, num_kv_heads)
        state.set_beta([beta], prompt_len=2, compacted_len=3,
                       total_kv_len=8, device=torch.device("cpu"))

        hook = _make_beta_hook(0, state)

        # Simulate attention_mask: (B=1, 1, Q=4, KV=8)
        mask = torch.zeros(1, 1, 4, 8)
        kwargs = {'attention_mask': mask, 'hidden_states': torch.zeros(1)}

        result = hook(None, (), kwargs)
        _, new_kwargs = result
        new_mask = new_kwargs['attention_mask']

        # Should be expanded to (1, 4, 4, 8)
        assert new_mask.shape == (1, num_heads, 4, 8)

        # Prompt and suffix positions should be 0
        assert (new_mask[:, :, :, :2] == 0).all()
        assert (new_mask[:, :, :, 5:] == 0).all()

        # Compacted positions [2:5] should have beta=1.0 (clamped)
        assert (new_mask[:, :, :, 2:5] == 1.0).all()

    def test_hook_noop_when_inactive(self):
        from prime_rl.trainer.rl.compaction import _BetaTrainingState, _make_beta_hook

        state = _BetaTrainingState(num_kv_heads=2, num_heads=4)
        hook = _make_beta_hook(0, state)

        mask = torch.zeros(1, 1, 4, 8)
        kwargs = {'attention_mask': mask}
        result = hook(None, (), kwargs)
        _, new_kwargs = result
        assert new_kwargs['attention_mask'] is mask  # same object, unmodified

    def test_hook_noop_when_mask_is_none(self):
        from prime_rl.trainer.rl.compaction import _BetaTrainingState, _make_beta_hook

        state = _BetaTrainingState(num_kv_heads=2, num_heads=4)
        state.active = True
        state.beta_per_layer[0] = torch.zeros(1, 4, 1, 8)
        hook = _make_beta_hook(0, state)

        kwargs = {'attention_mask': None}
        result = hook(None, (), kwargs)
        _, new_kwargs = result
        assert new_kwargs['attention_mask'] is None


# ── 4. Register/remove hooks ──────────────────────────────────────────────

class TestRegisterBetaHooks:
    def test_register_and_remove(self):
        from prime_rl.trainer.rl.compaction import (
            _BetaTrainingState, _register_beta_hooks, _find_attention_modules,
        )

        class FakeAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(4, 4)
                self.k_proj = torch.nn.Linear(4, 4)
                self.v_proj = torch.nn.Linear(4, 4)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn0 = FakeAttn()
                self.attn1 = FakeAttn()

        model = FakeModel()
        state = _BetaTrainingState(num_kv_heads=1, num_heads=1)
        hooks = _register_beta_hooks(model, state)

        assert len(hooks) == 2
        # Hooks are registered
        assert len(model.attn0._forward_pre_hooks) == 1
        assert len(model.attn1._forward_pre_hooks) == 1

        # Remove hooks
        for h in hooks:
            h.remove()
        assert len(model.attn0._forward_pre_hooks) == 0
        assert len(model.attn1._forward_pre_hooks) == 0


# ── 5. E2E: segmented_forward with/without beta ──────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestSegmentedForwardBeta:

    @pytest.fixture
    def small_model(self):
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

    def test_beta_changes_post_compaction_logits(self, small_model):
        """With compute_beta=True, post-compaction logits should differ from
        compute_beta=False (same KV contents, different attention computation)."""
        model, tokenizer = small_model

        prompt = "Solve: 2+2="
        text = prompt + " four. And 3+3= six. The answer is clearly obvious to everyone."
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

        from prime_rl.trainer.rl.compaction import segmented_forward

        model.config.use_cache = False

        with torch.no_grad():
            out_no_beta = segmented_forward(
                model=model,
                input_ids=input_ids,
                position_ids=position_ids,
                segment_boundaries=boundaries,
                prompt_len=prompt_len,
                compact_target_ratio=0.25,
                compact_window=None,
                temperature=temperature,
                compute_beta=False,
            )

        model.config.use_cache = False

        with torch.no_grad():
            out_beta = segmented_forward(
                model=model,
                input_ids=input_ids,
                position_ids=position_ids,
                segment_boundaries=boundaries,
                prompt_len=prompt_len,
                compact_target_ratio=0.25,
                compact_window=None,
                temperature=temperature,
                compute_beta=True,
            )

        logits_no_beta = out_no_beta["logits"]
        logits_beta = out_beta["logits"]

        assert logits_no_beta.shape == logits_beta.shape

        # First segment logits (before compaction) should be identical
        first_seg_end = prompt_len + mid - 1
        first_diff = (logits_no_beta[:, :first_seg_end] - logits_beta[:, :first_seg_end]).abs().max()
        assert first_diff < 1e-3, f"First segment logits differ by {first_diff}"

        # Second segment logits (after compaction) should differ due to beta
        second_diff = (logits_no_beta[:, first_seg_end:] - logits_beta[:, first_seg_end:]).abs().max()
        assert second_diff > 1e-4, \
            f"Beta should change post-compaction logits, but max diff is only {second_diff}"

    def test_beta_output_shape(self, small_model):
        """Beta-enabled segmented_forward produces correct output shape."""
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
        with torch.no_grad():
            out = segmented_forward(
                model=model,
                input_ids=input_ids,
                position_ids=position_ids,
                segment_boundaries=[mid, completion_len],
                prompt_len=prompt_len,
                compact_target_ratio=0.25,
                compact_window=None,
                temperature=temperature,
                compute_beta=True,
            )

        assert out["logits"].shape[1] == seq_len

    def test_beta_gradient_flow(self, small_model):
        """Gradients should flow through beta-corrected segmented_forward."""
        model, _ = small_model
        model.train()

        seq_len = 20
        input_ids = torch.randint(0, 1000, (1, seq_len), device="cuda")
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        temperature = torch.ones(1, seq_len, device="cuda")
        prompt_len = 5
        completion_len = seq_len - prompt_len
        mid = completion_len // 2

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
            compute_beta=True,
        )

        loss = out["logits"].sum()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().max() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradients flowed through beta-corrected segmented_forward"

    def test_hooks_cleaned_up(self, small_model):
        """Beta hooks should be fully removed after segmented_forward."""
        model, tokenizer = small_model
        from prime_rl.trainer.rl.compaction import segmented_forward, _find_attention_modules

        # Count hooks before
        attn_modules = _find_attention_modules(model)
        hooks_before = sum(len(m._forward_pre_hooks) for m in attn_modules)

        text = "One two three four five six seven eight nine ten eleven twelve"
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        prompt_len = 2
        completion_len = seq_len - prompt_len

        model.config.use_cache = False
        with torch.no_grad():
            segmented_forward(
                model=model,
                input_ids=input_ids,
                position_ids=torch.arange(seq_len, device="cuda").unsqueeze(0),
                segment_boundaries=[completion_len // 2, completion_len],
                prompt_len=prompt_len,
                compact_target_ratio=0.25,
                compact_window=None,
                temperature=torch.ones(1, seq_len, device="cuda"),
                compute_beta=True,
            )

        # Hooks should be fully cleaned up
        hooks_after = sum(len(m._forward_pre_hooks) for m in attn_modules)
        assert hooks_after == hooks_before, \
            f"Beta hooks leaked: {hooks_after - hooks_before} hooks remain"


# ── 6. Config auto-sync ────────────────────────────────────────────────────

class TestConfigAutoSync:
    def test_compute_beta_synced_from_env_args(self):
        """When env args set compute_beta=true, trainer.compute_beta should be True."""
        from prime_rl.configs.rl import RLConfig
        from prime_rl.utils.config import cli
        import tempfile, os

        toml_content = """
max_steps = 1
seq_len = 512
output_dir = "outputs/test-beta-sync"

[model]
name = "Qwen/Qwen2.5-0.5B"

[deployment]
num_train_gpus = 1
num_infer_gpus = 0

[trainer]
dist_timeout_seconds = 60

[trainer.model]
impl = "auto"

[trainer.loss]
type = "default"

[orchestrator]
batch_size = 8

[[orchestrator.env]]
id = "test_env"
name = "compaction-rg-mix"
args = { gym = "rg_mix_env", compute_beta = true }
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            f.flush()
            try:
                config = cli(RLConfig, args=["@", f.name])
                assert config.trainer.compute_beta is True, \
                    "compute_beta should be auto-synced from env args"
            finally:
                os.unlink(f.name)

    def test_compute_beta_not_synced_when_absent(self):
        """When env args don't set compute_beta, trainer.compute_beta stays False."""
        from prime_rl.configs.rl import RLConfig
        from prime_rl.utils.config import cli
        import tempfile, os

        toml_content = """
max_steps = 1
seq_len = 512
output_dir = "outputs/test-beta-nosync"

[model]
name = "Qwen/Qwen2.5-0.5B"

[deployment]
num_train_gpus = 1
num_infer_gpus = 0

[trainer]
dist_timeout_seconds = 60

[trainer.model]
impl = "auto"

[trainer.loss]
type = "default"

[orchestrator]
batch_size = 8

[[orchestrator.env]]
id = "test_env"
args = {}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            f.flush()
            try:
                config = cli(RLConfig, args=["@", f.name])
                assert config.trainer.compute_beta is False, \
                    "compute_beta should remain False when not in env args"
            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

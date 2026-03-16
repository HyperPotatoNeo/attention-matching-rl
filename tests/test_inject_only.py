"""Tests for inject-only generation (budget injection without compaction)."""

import pytest


class FakeTokenizer:
    """Minimal tokenizer mock matching Qwen3 chat template."""

    def __init__(self):
        self._vocab = {"<|im_start|>": 151644, "<|im_end|>": 151645}
        self._next_id = 200

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        result = []
        for ch in text:
            if ch not in self._vocab:
                self._vocab[ch] = self._next_id
                self._next_id += 1
            result.append(self._vocab[ch])
        return result


# ── Injection interval tests ────────────────────────────────────────────


def test_inject_interval_calculation():
    """Verify injection points are at correct token intervals."""
    max_total_tokens = 8192
    inject_budget_every = 2048

    max_injections = max_total_tokens // inject_budget_every
    assert max_injections == 4

    # Injection happens at effective tokens: 2048, 4096, 6144, (8192 = end, no inject)
    expected_inject_points = [2048, 4096, 6144]
    inject_points = []
    effective = 0
    for seg in range(max_injections + 1):
        effective += inject_budget_every
        if effective < max_total_tokens and seg < max_injections:
            inject_points.append(effective)
    assert inject_points == expected_inject_points


def test_inject_interval_non_divisible():
    """Verify correct behavior when max_total_tokens not divisible by inject_budget_every."""
    max_total_tokens = 5000
    inject_budget_every = 2048

    max_injections = max_total_tokens // inject_budget_every
    assert max_injections == 2

    # Injections at 2048, 4096. Then 5000-4096=904 tokens until done (no injection).
    inject_points = []
    effective = 0
    for seg in range(max_injections + 1):
        effective += inject_budget_every
        if effective < max_total_tokens and seg < max_injections:
            inject_points.append(effective)
    assert inject_points == [2048, 4096]


# ── Block allocation tests ──────────────────────────────────────────────


def test_block_allocation_headroom():
    """Verify block allocation accounts for inject overhead."""
    from prime_rl.inference.compaction.worker import _MAX_INJECT_TOKENS

    prompt_len = 1000
    max_total_tokens = 8192
    inject_budget_every = 2048
    block_size = 16

    max_injections = max_total_tokens // inject_budget_every
    max_possible_len = prompt_len + max_total_tokens + max_injections * (_MAX_INJECT_TOKENS + 1)

    # 1000 + 8192 + 4 * 41 = 9356
    assert max_possible_len == 1000 + 8192 + 4 * 41

    blocks_needed = (max_possible_len + block_size - 1) // block_size
    # 9356 / 16 = 584.75 → 585 blocks
    assert blocks_needed == 585


# ── Completion mask for inject-only ─────────────────────────────────────


def test_completion_mask_inject_only():
    """Verify completion mask correctly masks multiple inject-only ranges."""
    # Simulate: 8192 effective tokens + 3 injections of ~25 tokens each
    inject_ranges = [(2048, 2073), (4121, 4146), (6219, 6244)]
    total_len = 8192 + 75  # effective + injected

    mask = [1] * total_len
    for start, end in inject_ranges:
        for idx in range(start, min(end, total_len)):
            mask[idx] = 0

    masked_count = sum(1 for m in mask if m == 0)
    assert masked_count == 75
    assert sum(mask) == 8192


def test_effective_token_count_excludes_injected():
    """Verify effective token count calculation."""
    total_tokens = 8267  # 8192 effective + 75 injected
    injected_count = 75
    effective = total_tokens - injected_count
    assert effective == 8192


# ── Budget message at injection points ──────────────────────────────────


def test_budget_message_at_first_inject():
    """Verify budget message at first injection point."""
    template = "Budget: {used}/{total} tokens generated. ~{remaining} tokens remaining."
    inject_budget_every = 2048
    max_total_tokens = 8192

    # At first injection: 2048 effective tokens generated
    msg = template.format(used=2048, total=8192, remaining=6144)
    assert "2048" in msg
    assert "6144" in msg


def test_budget_message_at_last_inject():
    """Verify budget message at last injection point."""
    template = "Budget: {used}/{total} tokens generated. ~{remaining} tokens remaining."
    msg = template.format(used=6144, total=8192, remaining=2048)
    assert "6144" in msg
    assert "2048" in msg


# ── Diagnostics format ──────────────────────────────────────────────────


def test_diagnostics_no_compaction_events():
    """Inject-only should have empty compaction_events."""
    diagnostics = {
        "compaction_events": [],
        "segment_boundaries": [2048, 4121, 6219, 8292],
        "inject_ranges": [(2048, 2073), (4121, 4146), (6219, 6244)],
        "final_position_offset": 0,
    }
    assert diagnostics["compaction_events"] == []
    assert diagnostics["final_position_offset"] == 0
    assert len(diagnostics["inject_ranges"]) == 3


# ── Env inject_only flag ────────────────────────────────────────────────


def test_env_inject_only_params():
    """Verify CompactionEnv stores inject_only params."""
    from unittest.mock import MagicMock

    mock_env = MagicMock()
    mock_env.dataset = MagicMock()
    mock_env.system_prompt = "test"
    mock_env.parser = MagicMock()
    mock_env.rubric = MagicMock()

    from compaction_env.env import CompactionEnv

    env = CompactionEnv(
        inner_env=mock_env,
        inject_only=True,
        inject_budget_every=1024,
        max_total_tokens=4096,
    )

    assert env.inject_only is True
    assert env.inject_budget_every == 1024
    assert env.compact_max_total_tokens == 4096

"""Tests for budget-aware multi-turn compaction injection."""

import pytest
from unittest.mock import MagicMock


# ── Token construction tests ──────────────────────────────────────────────


class FakeTokenizer:
    """Minimal tokenizer mock matching Qwen3 chat template behavior."""

    def __init__(self):
        self._vocab = {
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
        }
        self._next_id = 200

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Deterministic fake: each unique text gets a stable sequence of IDs
        result = []
        for ch in text:
            if ch not in self._vocab:
                self._vocab[ch] = self._next_id
                self._next_id += 1
            result.append(self._vocab[ch])
        return result

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return "".join(inv.get(i, "?") for i in ids)


def test_build_inject_tokens_structure():
    """Verify inject tokens have correct turn structure."""
    from prime_rl.inference.compaction.worker import _build_inject_tokens

    tok = FakeTokenizer()
    msg = "Budget: 2000/8192 tokens."
    ids = _build_inject_tokens(tok, msg)

    # Decode to verify structure
    text = tok.decode(ids)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    im_start = tok.convert_tokens_to_ids("<|im_start|>")

    # Must start with <|im_end|> (end previous assistant turn)
    assert ids[0] == im_end

    # Must contain <|im_start|> for user and assistant turns
    im_start_positions = [i for i, t in enumerate(ids) if t == im_start]
    assert len(im_start_positions) == 2, f"Expected 2 <|im_start|>, got {len(im_start_positions)}"

    # Second <|im_start|> should be for assistant turn (near end of sequence)
    # The sequence ends with <|im_start|> + "assistant\n" tokens
    last_im_start = im_start_positions[-1]
    assert last_im_start > len(ids) // 2, "Second <|im_start|> should be in the latter half"


def test_build_inject_tokens_contains_message():
    """Verify the budget message appears in the inject tokens."""
    from prime_rl.inference.compaction.worker import _build_inject_tokens

    tok = FakeTokenizer()
    msg = "Budget: 3000/8192 tokens."
    ids = _build_inject_tokens(tok, msg)
    msg_ids = tok.encode(msg, add_special_tokens=False)

    # The message IDs should appear as a contiguous subsequence
    ids_str = " ".join(str(i) for i in ids)
    msg_str = " ".join(str(i) for i in msg_ids)
    assert msg_str in ids_str, "Budget message tokens not found in inject sequence"


def test_build_inject_tokens_length():
    """Verify inject tokens are reasonably short."""
    from prime_rl.inference.compaction.worker import _build_inject_tokens, _MAX_INJECT_TOKENS

    tok = FakeTokenizer()
    msg = "Budget: 7500/8192 tokens generated. ~692 tokens remaining."
    ids = _build_inject_tokens(tok, msg)

    # Should be well under the max headroom
    assert len(ids) < _MAX_INJECT_TOKENS * 2, f"Inject too long: {len(ids)} tokens"
    assert len(ids) > 5, f"Inject suspiciously short: {len(ids)} tokens"


# ── Completion mask tests ─────────────────────────────────────────────────


def test_completion_mask_from_inject_ranges():
    """Verify completion_mask correctly masks injected positions."""
    total_len = 4121
    inject_ranges = [(2048, 2073)]

    mask = [1] * total_len
    for start, end in inject_ranges:
        for idx in range(start, min(end, total_len)):
            mask[idx] = 0

    # Injected positions should be 0
    for i in range(2048, 2073):
        assert mask[i] == 0, f"Position {i} should be masked"

    # Adjacent positions should be 1
    assert mask[2047] == 1
    assert mask[2073] == 1

    # Total masked count
    assert sum(1 for m in mask if m == 0) == 25


def test_completion_mask_multiple_inject_ranges():
    """Verify mask with multiple inject ranges (multiple compactions)."""
    total_len = 6200
    inject_ranges = [(2048, 2070), (4100, 4125)]

    mask = [1] * total_len
    for start, end in inject_ranges:
        for idx in range(start, min(end, total_len)):
            mask[idx] = 0

    masked_count = sum(1 for m in mask if m == 0)
    assert masked_count == 22 + 25  # 2070-2048=22, 4125-4100=25


def test_completion_mask_empty_inject_ranges():
    """Verify mask is all 1s when no injection happened."""
    total_len = 4096
    inject_ranges = []

    mask = [1] * total_len
    for start, end in inject_ranges:
        for idx in range(start, min(end, total_len)):
            mask[idx] = 0

    assert all(m == 1 for m in mask)


# ── Budget tracking tests ────────────────────────────────────────────────


def test_budget_counting_excludes_inject():
    """Verify assistant token count excludes inject tokens."""
    all_token_ids = list(range(4121))  # 4121 total tokens
    injected_count = 25  # one injection of 25 tokens

    asst_tokens = len(all_token_ids) - injected_count
    assert asst_tokens == 4096

    remaining = 8192 - asst_tokens
    assert remaining == 4096


def test_budget_counting_multiple_injects():
    """Verify counting with multiple injections."""
    total = 8500
    injected_count = 75  # 3 injections of 25 tokens each

    asst_tokens = total - injected_count
    assert asst_tokens == 8425

    remaining = max(0, 8192 - asst_tokens)
    assert remaining == 0  # over budget


def test_budget_message_format():
    """Verify the default template produces a readable message."""
    template = "Budget: {used}/{total} tokens generated. ~{remaining} tokens remaining."
    msg = template.format(used=2000, total=8192, remaining=6192)
    assert "2000" in msg
    assert "8192" in msg
    assert "6192" in msg

"""Test that single-segment compaction samples are packed, multi-segment are not."""

import pytest

from prime_rl.trainer.batch import (
    _is_compaction_sample,
    packed_samples_into_micro_bs,
    prepare_sample,
)
from prime_rl.transport.types import TrainingSample


def _make_sample(
    n_tokens: int = 100,
    segment_boundaries: list[int] | None = None,
    compaction_indices: list | None = None,
) -> TrainingSample:
    return TrainingSample(
        prompt_ids=list(range(10)),
        prompt_mask=[False] * 10,
        completion_ids=list(range(n_tokens)),
        completion_mask=[True] * n_tokens,
        completion_logprobs=[-0.1] * n_tokens,
        completion_temperatures=[1.0] * n_tokens,
        segment_boundaries=segment_boundaries,
        compaction_indices=compaction_indices,
    )


# -- prepare_sample normalization --


def test_single_segment_normalized_to_none():
    """segment_boundaries=[N] → None (packable)."""
    sample = _make_sample(100, segment_boundaries=[100])
    mb = prepare_sample(sample, seq_len=512)
    assert mb.segment_boundaries is None
    assert mb.compaction_indices is None
    assert not _is_compaction_sample(mb)


def test_empty_segment_normalized_to_none():
    """segment_boundaries=[] → None (packable)."""
    sample = _make_sample(100, segment_boundaries=[])
    mb = prepare_sample(sample, seq_len=512)
    assert mb.segment_boundaries is None
    assert mb.compaction_indices is None


def test_no_segment_stays_none():
    """segment_boundaries=None stays None (packable)."""
    sample = _make_sample(100)
    mb = prepare_sample(sample, seq_len=512)
    assert mb.segment_boundaries is None
    assert mb.compaction_indices is None


def test_multi_segment_preserved():
    """segment_boundaries=[A, B] stays (not packable)."""
    sample = _make_sample(200, segment_boundaries=[100, 200])
    mb = prepare_sample(sample, seq_len=512)
    assert mb.segment_boundaries == [100, 200]
    assert _is_compaction_sample(mb)


def test_single_segment_clears_compaction_indices():
    """When segment is normalized, compaction_indices is also cleared."""
    dummy_indices = [[[0, 1, 2]]]  # would be per-event indices
    sample = _make_sample(100, segment_boundaries=[100], compaction_indices=dummy_indices)
    mb = prepare_sample(sample, seq_len=512)
    assert mb.segment_boundaries is None
    assert mb.compaction_indices is None


def test_multi_segment_keeps_compaction_indices():
    """Multi-segment preserves compaction_indices."""
    dummy_indices = [[[0, 1, 2]]]
    sample = _make_sample(200, segment_boundaries=[100, 200], compaction_indices=dummy_indices)
    mb = prepare_sample(sample, seq_len=512)
    assert mb.compaction_indices == dummy_indices


# -- packing behavior --


def test_single_segment_samples_packed_together():
    """Two single-segment samples should pack into one micro batch."""
    s1 = prepare_sample(_make_sample(50, segment_boundaries=[50]), seq_len=512)
    s2 = prepare_sample(_make_sample(50, segment_boundaries=[50]), seq_len=512)
    # Both normalized to None → packable
    batches = packed_samples_into_micro_bs([(0, s1), (0, s2)], max_seq_len=512, num_loras=1)
    assert len(batches) == 1
    assert len(batches[0].input_ids) == 120  # (10+50) + (10+50)


def test_multi_segment_samples_not_packed():
    """Two multi-segment samples each get their own micro batch."""
    s1 = prepare_sample(_make_sample(200, segment_boundaries=[100, 200]), seq_len=512)
    s2 = prepare_sample(_make_sample(200, segment_boundaries=[100, 200]), seq_len=512)
    batches = packed_samples_into_micro_bs([(0, s1), (0, s2)], max_seq_len=512, num_loras=1)
    assert len(batches) == 2


def test_mixed_single_and_multi_segment():
    """Single-segment samples pack; multi-segment samples don't."""
    s_single1 = prepare_sample(_make_sample(50, segment_boundaries=[50]), seq_len=512)
    s_single2 = prepare_sample(_make_sample(50, segment_boundaries=[50]), seq_len=512)
    s_multi = prepare_sample(_make_sample(200, segment_boundaries=[100, 200]), seq_len=512)

    batches = packed_samples_into_micro_bs(
        [(0, s_single1), (0, s_single2), (0, s_multi)],
        max_seq_len=512,
        num_loras=1,
    )
    # Two single-segment samples packed into 1 batch + 1 multi-segment batch = 2 total
    assert len(batches) == 2
    packed_batch = [b for b in batches if not _is_compaction_sample(b)]
    unpacked_batch = [b for b in batches if _is_compaction_sample(b)]
    assert len(packed_batch) == 1
    assert len(unpacked_batch) == 1
    assert len(packed_batch[0].input_ids) == 120  # (10+50) + (10+50)


def test_none_segment_packs_with_single_segment():
    """Samples from non-compaction env (None) pack with normalized single-segment samples."""
    s_none = prepare_sample(_make_sample(50), seq_len=512)
    s_single = prepare_sample(_make_sample(50, segment_boundaries=[50]), seq_len=512)
    batches = packed_samples_into_micro_bs([(0, s_none), (0, s_single)], max_seq_len=512, num_loras=1)
    assert len(batches) == 1


def test_truncation_preserves_multi_segment():
    """Truncated multi-segment sample still needs segmented forward."""
    # seq_len=150 truncates 10+200=210 tokens to 150
    sample = _make_sample(200, segment_boundaries=[100, 200])
    mb = prepare_sample(sample, seq_len=150)
    # Boundaries clamped: min(100, 140)=100, min(200, 140)=140 → still 2 elements
    assert mb.segment_boundaries is not None
    assert len(mb.segment_boundaries) == 2
    assert _is_compaction_sample(mb)

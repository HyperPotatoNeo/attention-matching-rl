import msgspec


# Orchestrator -> Packer
class TrainingSample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A single training example."""

    prompt_ids: list[int]
    prompt_mask: list[bool]
    completion_ids: list[int]
    completion_mask: list[bool]
    completion_logprobs: list[float]
    completion_temperatures: list[float]  # Per-token temperatures used during generation
    teacher_logprobs: list[float] | None = None
    advantage: float | None = None
    reward: float | None = None

    # Multimodal fields (Qwen3-VL) — pixel_values stored as raw float32 bytes for efficient serialization
    pixel_values: bytes | None = None
    pixel_values_shape: list[int] | None = None  # [num_patches, patch_dim]
    # image_grid_thw: grid dimensions [num_images, 3] where each entry is [temporal, height, width]
    image_grid_thw: list[list[int]] | None = None

    routed_experts: list[list[list[int]]] | None = None  # [seq_len, layers, topk]

    # Compaction training: cumulative completion token counts at end of each segment
    # e.g. [2048, 4096, 6144] means segment 0 had 2048 tokens, segment 1 had 2048 more, etc.
    segment_boundaries: list[int] | None = None

    # Compaction training: per-event top-k indices from inference for deterministic replay.
    # Shape: compaction_indices[event_idx][layer_idx] = list[list[int]] (H × target_len)
    compaction_indices: list | None = None

    # Compaction training: per-event compact_window used in inference.
    # Lets the trainer match the exact compaction range (important for turn-based mode
    # where protected turns narrow the window).
    compact_windows: list[int] | None = None


class TrainingBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A batch of training examples with metadata for transport."""

    examples: list[TrainingSample]
    step: int
    run_idx: int | None = None


# Packer -> Trainer
class MicroBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A micro batch of data for training."""

    input_ids: list[int]
    loss_mask: list[bool]
    advantages: list[float]
    inference_logprobs: list[float]
    position_ids: list[int]
    temperatures: list[float]  # Per-token temperatures used during generation
    teacher_logprobs: list[float] | None = None
    lora_num_tokens: list[int] | None = None
    routed_experts: list[list[list[int]]] | None = None

    # Compaction training: cumulative completion token counts at end of each segment
    segment_boundaries: list[int] | None = None

    # Compaction training: per-event top-k indices from inference
    compaction_indices: list | None = None

    # Compaction training: per-event compact_window from inference
    compact_windows: list[int] | None = None

    # Multimodal fields (Qwen3-VL) — pixel_values stored as raw float32 bytes for efficient serialization
    pixel_values: bytes | None = None
    pixel_values_shape: list[int] | None = None  # [num_patches, patch_dim]
    # image_grid_thw: grid dimensions [num_images, 3] where each entry is [temporal, height, width]
    image_grid_thw: list[list[int]] | None = None

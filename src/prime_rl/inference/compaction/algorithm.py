"""Attention Matching compaction algorithm (pure tensor operations, no vLLM dependency).

Used by both the inference worker (vLLM server-side) and the trainer (FSDP2 training loop)
to compact KV caches using the same algorithm.
"""

import math

import torch


def compact_kv(
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    prompt_len: int,
    target_ratio: float,
    num_kv_heads: int,
    head_size: int,
    device: torch.device,
    num_queries: int = 64,
    compact_window: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compact assistant KV prefix via Attention Matching (beta=0, random queries).

    When compact_window is set, only the first `compact_window` assistant tokens
    are compressed. The full assistant KV (including suffix beyond the window) is
    used for attention scoring so the algorithm sees the full context. The suffix
    KV is preserved unchanged by the caller.

    All heads are processed in a single batched operation per layer (no Python
    loop over heads). Random queries are shared across heads within a layer.

    Args:
        keys: Per-layer key tensors, each (seq_len, num_kv_heads, head_size)
        values: Per-layer value tensors, each (seq_len, num_kv_heads, head_size)
        prompt_len: Number of prompt tokens at the start
        target_ratio: Fraction of prefix keys to keep (e.g. 0.25 keeps 25%)
        num_kv_heads: Number of KV heads
        head_size: Dimension per head
        device: CUDA device
        num_queries: Number of random query probes
        compact_window: If set, only compress first N assistant tokens

    Returns:
        c1[layer]: (target_len, num_kv_heads, head_size) - compacted prefix keys
        c2[layer]: (target_len, num_kv_heads, head_size) - compacted prefix values
    """
    num_layers = len(keys)
    dtype = keys[0].dtype
    scale = 1.0 / math.sqrt(head_size)

    c1_list, c2_list = [], []

    for layer_idx in range(num_layers):
        asst_K = keys[layer_idx][prompt_len:]
        asst_V = values[layer_idx][prompt_len:]
        asst_len = asst_K.shape[0]
        window = min(compact_window or asst_len, asst_len)
        target_len = max(1, int(window * target_ratio))

        # (num_kv_heads, asst_len, head_size) — heads-first for batched ops
        K_h = asst_K.permute(1, 0, 2).float()
        V_h = asst_V.permute(1, 0, 2).float()
        Kp_h = K_h[:, :window, :]

        # Per-head random queries: (H, num_queries, head_size)
        Q = torch.randn(num_kv_heads, num_queries, head_size,
                         device=device, dtype=torch.float32)

        # Batched attention: (H, Q, T) = (H, Q, D) @ (H, D, T)
        full_scores = torch.bmm(Q, K_h.transpose(1, 2)) * scale
        full_attn = torch.softmax(full_scores, dim=-1)  # (H, Q, T)

        # Importance scores per prefix position: RMS attention weight
        prefix_attn = full_attn[:, :, :window]  # (H, Q, window)
        importance = prefix_attn.pow(2).mean(dim=1).sqrt()  # (H, window)

        # Top-k selection per head
        topk_indices = importance.topk(target_len, dim=-1).indices  # (H, target_len)
        topk_indices = topk_indices.sort(dim=-1).values

        # Gather selected keys: (H, target_len, D)
        idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, head_size)
        c1_h = torch.gather(Kp_h, 1, idx_expanded)

        # Batched lstsq for C2 values
        # Y = prefix_attn @ V_prefix: (H, Q, D)
        Vp_h = V_h[:, :window, :]
        Y = torch.bmm(prefix_attn, Vp_h)

        # X = softmax(Q @ C1^T / sqrt(d)): (H, Q, target_len)
        X = torch.bmm(Q, c1_h.transpose(1, 2)) * scale
        X = torch.softmax(X, dim=-1)

        # Solve: X @ C2 ≈ Y → C2 = lstsq(X, Y)
        c2_h = torch.linalg.lstsq(X, Y).solution  # (H, target_len, D)

        # Clamp C2 to prevent extreme values from ill-conditioned lstsq.
        # Use the original value range as reference — C2 should be in the
        # same ballpark as the original values it replaces.
        v_absmax = Vp_h.abs().max().item() * 2.0 + 1.0
        c2_h = c2_h.clamp(-v_absmax, v_absmax).to(dtype)  # (H, target_len, D)

        # Back to (target_len, H, D) format
        c1_out = torch.gather(
            asst_K[:window],  # (window, H, D) in original dtype
            0,
            topk_indices.permute(1, 0).unsqueeze(-1).expand(-1, -1, head_size),
        )
        c1_list.append(c1_out)
        c2_list.append(c2_h.permute(1, 0, 2).to(dtype))

    return c1_list, c2_list

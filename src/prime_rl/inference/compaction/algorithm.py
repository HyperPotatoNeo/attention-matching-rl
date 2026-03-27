"""Attention Matching compaction algorithm (pure tensor operations, no vLLM dependency).

Used by both the inference worker (vLLM server-side) and the trainer (FSDP2 training loop)
to compact KV caches using the same algorithm.
"""

import math

import torch


def _solve_beta_nnls(
    Q: torch.Tensor,
    K_full: torch.Tensor,
    C1: torch.Tensor,
    scale: float,
    iters: int = 50,
) -> torch.Tensor:
    """Solve for per-key bias beta via non-negative least squares (NNLS).

    Beta corrects the partition function mismatch between full and compacted
    attention: find B >= 0 such that M @ B ≈ target, then beta = log(B).

    Where:
        target_i = sum_k exp(q_i @ k_k / sqrt(d))    -- full partition
        M_{i,j}  = exp(q_i @ c1_j / sqrt(d))         -- compacted terms

    Args:
        Q: (H, n, D) random queries
        K_full: (H, T, D) full keys (all positions, for partition function)
        C1: (H, t, D) selected compacted keys
        scale: 1/sqrt(head_dim)
        iters: projected gradient descent iterations

    Returns:
        beta: (H, t) per-head per-key bias in log-space
    """
    H, n, D = Q.shape
    t = C1.shape[1]

    # Full partition function: target = sum_k exp(q @ k / sqrt(d))  per query
    full_scores = torch.bmm(Q, K_full.transpose(1, 2)) * scale  # (H, n, T)
    full_max = full_scores.max(dim=-1, keepdim=True).values  # stability
    target = torch.exp(full_scores - full_max).sum(dim=-1)  # (H, n)

    # Design matrix: M_{i,j} = exp(q_i @ c1_j / sqrt(d))
    comp_scores = torch.bmm(Q, C1.transpose(1, 2)) * scale  # (H, n, t)
    M = torch.exp(comp_scores - full_max)  # (H, n, t) — same shift for consistency

    # Projected gradient descent for NNLS: min ||M @ B - target||^2, B >= 0
    # Initialize B = target.mean() / (M.mean() * t) — rough uniform estimate
    B = torch.full((H, t), 1.0, device=Q.device, dtype=torch.float32)

    # Step size: 1 / ||M||^2 (spectral norm squared, approximated per head)
    MtM_diag = (M * M).sum(dim=1)  # (H, t) — diagonal of M^T M
    step_size = 1.0 / (MtM_diag.max(dim=-1, keepdim=True).values + 1e-8)  # (H, 1)

    for _ in range(iters):
        residual = torch.bmm(M.transpose(1, 2), (torch.bmm(M, B.unsqueeze(-1)).squeeze(-1) - target).unsqueeze(-1)).squeeze(-1)
        # residual = M^T @ (M @ B - target), shape (H, t)
        B = (B - step_size * residual).clamp(min=1e-12)

    beta = torch.log(B.clamp(min=1e-10))  # (H, t)
    return beta


def compact_kv_range(
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    compact_start: int,
    compact_end: int,
    target_ratio: float,
    num_kv_heads: int,
    head_size: int,
    device: torch.device,
    num_queries: int = 64,
    compute_beta: bool = False,
    beta_nnls_iters: int = 50,
    suffix_queries: list[torch.Tensor] | None = None,
    seed: int | None = None,
    forced_indices: list[torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None, list[torch.Tensor]]:
    """Compact a specific range of KV cache via Attention Matching.

    Compresses keys/values in [compact_start, compact_end). The full KV
    (including tokens outside the range) is used for attention scoring so the
    algorithm sees the full context.

    Args:
        keys: Per-layer key tensors, each (seq_len, num_kv_heads, head_size)
        values: Per-layer value tensors, each (seq_len, num_kv_heads, head_size)
        compact_start: Start index of the region to compress
        compact_end: End index of the region to compress (exclusive)
        target_ratio: Fraction of region keys to keep (e.g. 0.25 keeps 25%)
        num_kv_heads: Number of KV heads
        head_size: Dimension per head
        device: CUDA device
        num_queries: Number of random query probes (used only when suffix_queries is None)
        compute_beta: If True, compute NNLS beta bias for partition function
            correction. Returns per-layer beta tensors as the third element.
        beta_nnls_iters: Number of projected gradient descent iterations for NNLS.
        suffix_queries: Per-layer query tensors (num_kv_heads, num_q, head_size). When
            provided, used instead of random Gaussian probes for importance scoring.
        seed: RNG seed for reproducible random probes (used only when suffix_queries is None).
        forced_indices: Per-layer (num_kv_heads, target_len) index tensors. When provided,
            skips importance scoring and uses these indices directly for key selection.

    Returns:
        c1[layer]: (target_len, num_kv_heads, head_size) - compacted keys
        c2[layer]: (target_len, num_kv_heads, head_size) - compacted values
        beta[layer] or None: (target_len, num_kv_heads) - per-key bias if compute_beta
        indices[layer]: (num_kv_heads, target_len) - selected top-k indices per layer
    """
    num_layers = len(keys)
    dtype = keys[0].dtype
    scale = 1.0 / math.sqrt(head_size)
    region_len = compact_end - compact_start

    c1_list, c2_list = [], []
    beta_list = [] if compute_beta else None
    indices_list = []

    rng = None
    if suffix_queries is None and seed is not None:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

    for layer_idx in range(num_layers):
        all_K = keys[layer_idx]
        all_V = values[layer_idx]

        region_K = all_K[compact_start:compact_end]
        region_V = all_V[compact_start:compact_end]
        target_len = max(1, int(region_len * target_ratio))

        # (H, seq_len, D) and (H, region_len, D) — heads-first for batched ops
        K_h = all_K.permute(1, 0, 2).float()
        V_h = all_V.permute(1, 0, 2).float()
        Rk_h = K_h[:, compact_start:compact_end, :]
        Rv_h = V_h[:, compact_start:compact_end, :]

        if suffix_queries is not None and suffix_queries[layer_idx].shape[1] > 0:
            Q = suffix_queries[layer_idx].float()
        elif rng is not None:
            Q = torch.randn(num_kv_heads, num_queries, head_size,
                            device=device, dtype=torch.float32, generator=rng)
        else:
            Q = torch.randn(num_kv_heads, num_queries, head_size,
                            device=device, dtype=torch.float32)

        # Full attention over all keys: (H, Q, seq_len)
        full_scores = torch.bmm(Q, K_h.transpose(1, 2)) * scale
        full_attn = torch.softmax(full_scores, dim=-1)

        # Importance over region only
        region_attn = full_attn[:, :, compact_start:compact_end]  # (H, Q, region_len)

        if forced_indices is not None:
            topk_indices = forced_indices[layer_idx].to(device=device, dtype=torch.int64)
            topk_indices = topk_indices.sort(dim=-1).values
        else:
            importance = region_attn.pow(2).mean(dim=1).sqrt()  # (H, region_len)
            topk_indices = importance.topk(target_len, dim=-1).indices
            topk_indices = topk_indices.sort(dim=-1).values

        indices_list.append(topk_indices)

        # Gather selected keys: (H, target_len, D)
        idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, head_size)
        c1_h = torch.gather(Rk_h, 1, idx_expanded)

        # Beta computation
        layer_beta = None
        if compute_beta:
            layer_beta = _solve_beta_nnls(Q, Rk_h, c1_h, scale, iters=beta_nnls_iters)

        # lstsq for C2: Y = region_attn @ V_region
        Y = torch.bmm(region_attn, Rv_h)

        X = torch.bmm(Q, c1_h.transpose(1, 2)) * scale
        if layer_beta is not None:
            X = X + layer_beta.unsqueeze(1)
        X = torch.softmax(X, dim=-1)

        # Ridge regression: (X^T X + λI) c2 = X^T Y
        # More robust than lstsq on CUDA which throws on rank-deficient matrices.
        XtX = torch.bmm(X.transpose(1, 2), X)
        XtX.diagonal(dim1=-2, dim2=-1).add_(1e-3)
        c2_h = torch.linalg.solve(XtX, torch.bmm(X.transpose(1, 2), Y))

        v_absmax = Rv_h.abs().max().item() * 2.0 + 1.0
        c2_h = c2_h.clamp(-v_absmax, v_absmax).to(dtype)

        # Back to (target_len, H, D) format
        c1_out = torch.gather(
            region_K,
            0,
            topk_indices.permute(1, 0).unsqueeze(-1).expand(-1, -1, head_size),
        )
        c1_list.append(c1_out)
        c2_list.append(c2_h.permute(1, 0, 2).to(dtype))

        if compute_beta and layer_beta is not None:
            layer_beta = torch.nan_to_num(layer_beta, nan=0.0)
            beta_list.append(layer_beta.permute(1, 0).to(torch.float32))

    return c1_list, c2_list, beta_list, indices_list


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
    compute_beta: bool = False,
    beta_nnls_iters: int = 50,
    suffix_queries: list[torch.Tensor] | None = None,
    seed: int | None = None,
    forced_indices: list[torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None, list[torch.Tensor]]:
    """Compact assistant KV prefix via Attention Matching.

    Delegates to compact_kv_range with compact_start=prompt_len and
    compact_end=prompt_len + window.

    Args:
        keys: Per-layer key tensors, each (seq_len, num_kv_heads, head_size)
        values: Per-layer value tensors, each (seq_len, num_kv_heads, head_size)
        prompt_len: Number of prompt tokens at the start
        target_ratio: Fraction of prefix keys to keep (e.g. 0.25 keeps 25%)
        num_kv_heads: Number of KV heads
        head_size: Dimension per head
        device: CUDA device
        num_queries: Number of random query probes (used only when suffix_queries is None)
        compact_window: If set, only compress first N assistant tokens
        compute_beta: If True, compute NNLS beta bias for partition function
            correction. Returns per-layer beta tensors as the third element.
        beta_nnls_iters: Number of projected gradient descent iterations for NNLS.
        suffix_queries: Per-layer query tensors (num_kv_heads, num_q, head_size). When
            provided, used instead of random Gaussian probes for importance scoring.
        seed: RNG seed for reproducible random probes (used only when suffix_queries is None).
        forced_indices: Per-layer (num_kv_heads, target_len) index tensors. When provided,
            skips importance scoring and uses these indices directly for key selection.

    Returns:
        c1[layer]: (target_len, num_kv_heads, head_size) - compacted prefix keys
        c2[layer]: (target_len, num_kv_heads, head_size) - compacted prefix values
        beta[layer] or None: (target_len, num_kv_heads) - per-key bias if compute_beta
        indices[layer]: (num_kv_heads, target_len) - selected top-k indices per layer
    """
    asst_len = keys[0].shape[0] - prompt_len
    window = min(compact_window or asst_len, asst_len)
    return compact_kv_range(
        keys, values,
        compact_start=prompt_len,
        compact_end=prompt_len + window,
        target_ratio=target_ratio,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        device=device,
        num_queries=num_queries,
        compute_beta=compute_beta,
        beta_nnls_iters=beta_nnls_iters,
        suffix_queries=suffix_queries,
        seed=seed,
        forced_indices=forced_indices,
    )

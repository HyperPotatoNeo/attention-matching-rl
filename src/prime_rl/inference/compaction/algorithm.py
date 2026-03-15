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


def compact_kv(
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    prompt_len: int,
    target_ratio: float,
    num_kv_heads: int,
    head_size: int,
    device: torch.device,
    num_queries: int = 1024,
    compact_window: int | None = None,
    compute_beta: bool = False,
    beta_nnls_iters: int = 50,
    seed: int = 0,
    suffix_queries: list[torch.Tensor] | None = None,
    forced_indices: list[torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None, list[torch.Tensor]]:
    """Compact assistant KV prefix via Attention Matching.

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
        compute_beta: If True, compute NNLS beta bias for partition function
            correction. Returns per-layer beta tensors as the third element.
        beta_nnls_iters: Number of projected gradient descent iterations for NNLS.
        seed: Deterministic random query seed. Seeded per-layer as
            seed + layer_idx. Ensures identical compaction between inference
            and training replay.
        suffix_queries: Per-layer suffix query tensors for importance scoring.
            Each tensor has shape (num_kv_heads, num_suffix_tokens, head_size)
            in float32. When provided, these real queries from the suffix tokens
            replace seeded random probes, giving exact importance scores.
            Falls back to random queries if None or if a layer's tensor is empty.
        forced_indices: Per-layer pre-computed top-k indices from inference.
            Each tensor has shape (num_kv_heads, target_len) with int64 dtype.
            When provided, skips importance scoring and top-k selection entirely.
            C2 is still recomputed using the trainer's own KV cache for correct
            gradients. Used to guarantee identical key selection between inference
            and training when suffix queries produce numerical differences.
    Returns:
        c1[layer]: (target_len, num_kv_heads, head_size) - compacted prefix keys
        c2[layer]: (target_len, num_kv_heads, head_size) - compacted prefix values
        beta[layer] or None: (target_len, num_kv_heads) - per-key bias if compute_beta
        indices[layer]: (num_kv_heads, target_len) - selected window-relative indices
    """
    num_layers = len(keys)
    dtype = keys[0].dtype
    scale = 1.0 / math.sqrt(head_size)

    c1_list, c2_list = [], []
    beta_list = [] if compute_beta else None
    indices_list = []

    for layer_idx in range(num_layers):
        all_K = keys[layer_idx]  # (seq_len, H, D) — prompt + assistant
        all_V = values[layer_idx]
        seq_len = all_K.shape[0]
        asst_len = seq_len - prompt_len
        window = min(compact_window or asst_len, asst_len)
        target_len = max(1, int(window * target_ratio))

        window_start = prompt_len
        window_end = prompt_len + window
        suffix_len = asst_len - window

        # (H, seq_len, D) — heads-first for batched ops
        K_all_h = all_K.permute(1, 0, 2).float()
        V_all_h = all_V.permute(1, 0, 2).float()
        Kw_h = K_all_h[:, window_start:window_end, :]  # (H, window, D)

        # Query probes: use suffix queries if available, else seeded random
        if suffix_queries is not None and suffix_queries[layer_idx].shape[1] > 0:
            Q = suffix_queries[layer_idx].to(device=device, dtype=torch.float32)
        else:
            g = torch.Generator(device=device)
            g.manual_seed(seed + layer_idx)
            Q = torch.randn(num_kv_heads, num_queries, head_size,
                             device=device, dtype=torch.float32, generator=g)

        # Attention over ALL keys (prompt + assistant) for full-context scoring
        full_scores = torch.bmm(Q, K_all_h.transpose(1, 2)) * scale  # (H, Q, seq_len)
        full_attn = torch.softmax(full_scores, dim=-1)

        # Window attention from full-context softmax (used for importance + C2 target)
        window_attn = full_attn[:, :, window_start:window_end]  # (H, Q, window)

        # Top-k selection: use forced indices or compute from importance scores
        if forced_indices is not None:
            topk_indices = forced_indices[layer_idx].to(device)  # (H, target_len)
        else:
            importance = window_attn.pow(2).mean(dim=1).sqrt()  # (H, window)
            topk_indices = importance.topk(target_len, dim=-1).indices  # (H, target_len)
            topk_indices = topk_indices.sort(dim=-1).values

        # Clamp all indices to window bounds — inject tokens can shift the
        # effective window between inference and training replay, and async
        # CUDA errors from OOB gathers are hard to diagnose.
        topk_indices = topk_indices.clamp(min=0, max=max(window - 1, 0))

        indices_list.append(topk_indices)

        # Gather selected keys: (H, target_len, D)
        idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, head_size)
        idx_clamped = idx_expanded.clamp(min=0, max=max(Kw_h.shape[1] - 1, 0))
        c1_h = torch.gather(Kw_h, 1, idx_clamped)

        # Compute beta if requested (full keys for correct partition function)
        layer_beta = None
        if compute_beta:
            layer_beta = _solve_beta_nnls(Q, K_all_h, c1_h, scale, iters=beta_nnls_iters)

        # C2 target: window's contribution to full-context output.
        # full_attn is conditioned on prompt (prompt keys are in the softmax
        # denominator), so window positions that compete with strong prompt
        # keys get appropriately lower weights. But we do NOT subtract
        # prompt/suffix corrections — that embeds "anti-prompt" bias into C2
        # which compounds across multiple compactions.
        c1_scores = torch.bmm(Q, c1_h.transpose(1, 2)) * scale
        if layer_beta is not None:
            c1_scores = c1_scores + layer_beta.unsqueeze(1)

        Vw_h = V_all_h[:, window_start:window_end, :]
        Y = torch.bmm(window_attn, Vw_h)  # (H, Q, D)

        X = torch.softmax(c1_scores, dim=-1)  # (H, Q, target_len)

        # Solve C2 via ridge regression: (X^T X + λI) C2 = X^T Y
        # lstsq's gels driver (CUDA) assumes full-rank X and crashes when
        # special tokens create peaked attention → near-rank-deficient X.
        # Ridge regression handles arbitrary conditioning by construction.
        lambda_reg = 1e-4
        XtX = torch.bmm(X.transpose(1, 2), X)  # (H, target_len, target_len)
        XtX.diagonal(dim1=-2, dim2=-1).add_(lambda_reg)
        XtY = torch.bmm(X.transpose(1, 2), Y)  # (H, target_len, D)
        c2_h = torch.linalg.solve(XtX, XtY).to(dtype)  # (H, target_len, D)

        # Back to (target_len, H, D) format
        window_K = all_K[window_start:window_end]
        c1_idx = topk_indices.permute(1, 0).unsqueeze(-1).expand(-1, -1, head_size)
        c1_idx = c1_idx.clamp(max=window_K.shape[0] - 1)
        c1_out = torch.gather(window_K, 0, c1_idx)
        c1_list.append(c1_out)
        c2_list.append(c2_h.permute(1, 0, 2).to(dtype))

        if compute_beta and layer_beta is not None:
            beta_list.append(layer_beta.permute(1, 0).to(torch.float32))

    return c1_list, c2_list, beta_list, indices_list

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
    num_queries: int = 64,
    compact_window: int | None = None,
    compute_beta: bool = False,
    beta_nnls_iters: int = 50,
    seed: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None]:
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
        seed: If set, use deterministic random queries seeded per-layer as
            seed + layer_idx. Ensures identical compaction between inference
            and training replay.

    Returns:
        c1[layer]: (target_len, num_kv_heads, head_size) - compacted prefix keys
        c2[layer]: (target_len, num_kv_heads, head_size) - compacted prefix values
        beta[layer] or None: (target_len, num_kv_heads) - per-key bias if compute_beta
    """
    num_layers = len(keys)
    dtype = keys[0].dtype
    scale = 1.0 / math.sqrt(head_size)

    c1_list, c2_list = [], []
    beta_list = [] if compute_beta else None

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
        # Deterministic seeding ensures identical compaction in inference and training
        if seed is not None:
            g = torch.Generator(device=device)
            g.manual_seed(seed + layer_idx)
            Q = torch.randn(num_kv_heads, num_queries, head_size,
                             device=device, dtype=torch.float32, generator=g)
        else:
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

        # Compute beta if requested
        layer_beta = None
        if compute_beta:
            # Use ALL assistant keys (window + suffix) for the target partition
            # function. During decode, softmax runs over prompt + C1+beta + suffix
            # + decoded tokens. Using only window keys produces beta that's too
            # small, causing the model to under-attend to compacted context.
            # The larger beta from full keys keeps compacted keys relevant.
            # NaN safety nets (lines below) handle numerical edge cases.
            layer_beta = _solve_beta_nnls(Q, K_h, c1_h, scale, iters=beta_nnls_iters)
            # layer_beta: (H, target_len)

        # Batched lstsq for C2 values
        # Y = prefix_attn @ V_prefix: (H, Q, D)
        Vp_h = V_h[:, :window, :]
        Y = torch.bmm(prefix_attn, Vp_h)

        # X = softmax(Q @ C1^T / sqrt(d) [+ beta]): (H, Q, target_len)
        X = torch.bmm(Q, c1_h.transpose(1, 2)) * scale
        if layer_beta is not None:
            X = X + layer_beta.unsqueeze(1)  # broadcast beta across queries
        X = torch.softmax(X, dim=-1)

        # Solve: X @ C2 ≈ Y → C2 = lstsq(X, Y)
        c2_h = torch.linalg.lstsq(X, Y).solution  # (H, target_len, D)

        # lstsq produces NaN when X is rank-deficient (e.g., peaked softmax
        # from large beta). Replace NaN entries with original values at
        # selected positions — equivalent to no value optimization for those.
        nan_mask = torch.isnan(c2_h)
        if nan_mask.any():
            c2_fallback = torch.gather(Vp_h, 1, idx_expanded)
            c2_h = torch.where(nan_mask, c2_fallback, c2_h)

        # Clamp C2 to prevent extreme values from ill-conditioned lstsq.
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

        if compute_beta and layer_beta is not None:
            # (target_len, H) for consistency with C1/C2 shape convention
            layer_beta = torch.nan_to_num(layer_beta, nan=0.0)
            beta_list.append(layer_beta.permute(1, 0).to(torch.float32))

    return c1_list, c2_list, beta_list

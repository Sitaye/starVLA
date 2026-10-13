import torch
import torch.nn.functional as F


def token_motion_loss(pred_flow: torch.Tensor, target_flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token motion loss: L_ij = 1/2 ||f_hat_ij - f*_ij||_2^2 over the 2 flow
    channels, kept per-token for spatial credit assignment.

    Anchor validity comes from the target: rows whose future frame fell outside
    the episode (no RAFT supervision) are stored as NaN and report valid=False
    with a zero loss contribution. The target is sanitized with nan_to_num before
    differencing, so invalid positions cannot inject 0 * NaN = NaN gradients into
    the Motion Head.

    :param pred_flow: torch.Tensor, [B, V, H_tok, W_tok, 2].
    :param target_flow: torch.Tensor, [B, V, H_tok, W_tok, 2]; NaN marks invalid anchors.
    :return: (per_token_loss [B, V, H_tok, W_tok], valid [B, V, H_tok, W_tok] bool).
    """
    valid = target_flow.isfinite().all(dim=-1)
    loss = 0.5 * ((pred_flow - torch.nan_to_num(target_flow)) ** 2).sum(dim=-1)
    return loss * valid, valid


def motion_credit(
    g_A_tok: torch.Tensor,
    g_M_tok: torch.Tensor,
    valid: torch.Tensor,
    q_map: torch.Tensor,
    w_mode: str = "compat",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rank-based compatibility credit assignment for motion supervision
    (docs/research/new_idea3.md section 4.3).

    C_ij = cos_fp32(g_A_ij, g_M_ij): per-token alignment between the action
           gradient and the motion gradient at the representation coordinate,
           or 0 when either gradient vanishes at that token.
    r_ij = ranknorm(C) over all valid tokens of the microbatch (cross-sample,
           cross-view), so r in [0, 1] and the mapping no longer collapses to
           near-uniform when the cosine distribution is narrow.
    w_ij = 0.5 + r_ij at valid tokens, bounded in [0.5, 1.5] and 0 elsewhere;
           the overall scale is fixed downstream by the a_i = m q w product
           renormalisation of the weighted motion loss.

    w_mode="shuffled" permutes the w values within each (sample, view) among
    valid tokens falling in the same fixed q stratum (the q grid [0.25, 1]
    split into 4 equal bins), keeping the weight distribution and the
    confidence strata intact while breaking the token correspondence; it
    still consumes the real gradients, and compat is reported pre-permutation.

    :param g_A_tok: torch.Tensor [B, V, H_tok, W_tok, D], action-loss gradient at visual tokens.
    :param g_M_tok: torch.Tensor [B, V, H_tok, W_tok, D], motion-loss gradient at visual tokens.
    :param valid: torch.Tensor, [B, V, H_tok, W_tok] bool, anchors with RAFT supervision.
    :param q_map: torch.Tensor, [B, V, H_tok, W_tok], raw q_i confidence in [0.25, 1]; strata only.
    :param w_mode: str, one of {"compat", "shuffled"}.
    :return: (w [B, V, H_tok, W_tok] detached, compat [B, V, H_tok, W_tok] detached).
    """
    g_A = g_A_tok.float()
    g_M = g_M_tok.float()
    compat = F.cosine_similarity(g_A, g_M, dim=-1)  # [B, V, H_tok, W_tok]
    compat = torch.where((g_A.norm(dim=-1) > 0) & (g_M.norm(dim=-1) > 0), compat, torch.zeros_like(compat))
    valid_flat = valid.reshape(-1)
    c_valid = compat.reshape(-1)[valid_flat]
    order = c_valid.argsort()
    ranks = torch.empty(c_valid.numel(), device=c_valid.device, dtype=c_valid.dtype)
    ranks[order] = torch.arange(c_valid.numel(), device=c_valid.device, dtype=c_valid.dtype)
    w_valid = 0.5 + ranks / max(c_valid.numel() - 1, 1)
    w_flat = torch.zeros(valid.numel(), device=c_valid.device, dtype=c_valid.dtype)
    w_flat[valid_flat] = w_valid
    w = w_flat.reshape(valid.shape)
    if w_mode == "shuffled":
        w = _shuffle_within_q_strata(w, q_map, valid)
    return w.detach(), compat.detach()


def _shuffle_within_q_strata(
    w: torch.Tensor, q_map: torch.Tensor, valid: torch.Tensor, num_strata: int = 4
) -> torch.Tensor:
    """
    Break the w <-> token correspondence while preserving its distribution: within
    each (sample, view) and each fixed stratum of the q grid [0.25, 1], permute the
    valid tokens' w values bijectively. Tokens outside the valid anchors keep w = 0.

    :param w: torch.Tensor, [B, V, H_tok, W_tok], detached credit weights.
    :param q_map: torch.Tensor, [B, V, H_tok, W_tok], raw q_i confidence in [0.25, 1].
    :param valid: torch.Tensor, [B, V, H_tok, W_tok] bool, anchors with RAFT supervision.
    :param num_strata: int, fixed equal-width bins on the q grid.
    :return: torch.Tensor, [B, V, H_tok, W_tok] permuted credit weights.
    """
    out = w.clone()
    edges = torch.linspace(0.25, 1.0, num_strata + 1, device=q_map.device)
    for b in range(valid.shape[0]):
        for v in range(valid.shape[1]):
            view_q = q_map[b, v]
            for k in range(num_strata):
                upper = edges[k + 1] if k < num_strata - 1 else edges[k + 1] + 1.0
                sel = valid[b, v] & (view_q >= edges[k]) & (view_q < upper)
                idx = sel.nonzero(as_tuple=True)
                if idx[0].numel() < 2:
                    continue
                vals = out[b, v][idx]
                out[b, v][idx[0], idx[1]] = vals[torch.randperm(vals.numel(), device=vals.device)]
    return out

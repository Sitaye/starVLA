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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rank-based compatibility credit assignment for motion supervision
    (docs/research/new_idea3.md section 4.3).

    C_ij = cos_fp32(g_A_ij, g_M_ij): per-token alignment between the action
           gradient and the motion gradient at the representation coordinate.
    Tokens whose motion target is valid but whose gradient norm is at the fp32
    noise floor carry no interpretable cosine direction; they are excluded from
    the ranking and keep the neutral weight w = 1 instead of being pushed to a
    mid-range C = 0 score.
    r_ij = AVERAGE ranknorm(C) over the rankable tokens of the SAME (sample, view):
           tied scores share the mean ordinal rank, so identical C maps to identical
           weights, an all-equal C maps every w to 1, and a single rankable token
           maps to 1. Ranking inside each view keeps every sample/view's motion-
           supervision budget local (no cross-sample cross-view credit transfer).
    w_ij = 0.5 + r_ij at rankable tokens, 1 at the remaining valid tokens, and 0
           at invalid anchors; the overall scale is fixed downstream by the
           per-(sample, view) budget renormalisation of a_i = m q w.

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
    :return: (w [B, V, H_tok, W_tok] detached, compat [B, V, H_tok, W_tok] detached,
             grad_valid_frac scalar detached = share of valid anchors that are rankable,
             changed_fraction scalar detached = share of valid anchors whose w the shuffle moved (0 in compat mode),
             weight_l1 scalar detached = mean |Δw| of the shuffle (0 in compat mode)).
    """
    g_A = g_A_tok.float()
    g_M = g_M_tok.float()
    compat = F.cosine_similarity(g_A, g_M, dim=-1).detach()  # [B, V, H_tok, W_tok]
    rankable = valid & (g_A.norm(dim=-1) > 1e-6) & (g_M.norm(dim=-1) > 1e-6)
    w = valid.to(compat.dtype)
    for b in range(valid.shape[0]):
        for v in range(valid.shape[1]):
            idx = rankable[b, v].nonzero(as_tuple=True)
            if idx[0].numel() == 0:
                continue
            c = compat[b, v][idx]
            n = c.numel()
            if n == 1:
                w[b, v][idx] = 1.0
                continue
            order = c.argsort()
            sorted_c = c[order]
            # Ties share the mean of their ordinal ranks, so equal scores get equal weights
            # and an all-equal c maps every weight to the mid-range value 1.
            group_id = torch.cat([sorted_c.new_zeros(1), (sorted_c[1:] != sorted_c[:-1]).cumsum(0)]).long()
            counts = torch.bincount(group_id)
            sums = torch.zeros(counts.numel(), device=c.device, dtype=c.dtype)
            sums.index_add_(0, group_id, torch.arange(n, device=c.device, dtype=c.dtype))
            ranks = torch.empty(n, device=c.device, dtype=c.dtype)
            ranks[order] = (sums / counts.to(c.dtype))[group_id]
            w[b, v][idx] = 0.5 + ranks / (n - 1)
    changed_fraction = w.new_zeros(())
    weight_l1 = w.new_zeros(())
    if w_mode == "shuffled":
        w_shuf = _shuffle_within_q_strata(w, q_map, valid)
        delta = (w_shuf - w)[valid]
        changed_fraction = (delta != 0).sum().float() / valid.sum().float().clamp(min=1)
        weight_l1 = delta.abs().sum().float() / valid.sum().float().clamp(min=1)
        w = w_shuf
    grad_valid_frac = rankable.sum().float() / valid.sum().float().clamp(min=1)
    return (
        w.detach(),
        compat.detach(),
        grad_valid_frac.detach(),
        changed_fraction.detach(),
        weight_l1.detach(),
    )


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

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
    per_token_motion_loss: torch.Tensor,
    g_A_tok: torch.Tensor | None,
    g_M_tok: torch.Tensor | None,
    valid: torch.Tensor,
    w_mode: str = "compat",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compatibility-aware credit assignment for motion supervision.

    C_ij  = cos_fp32(g_A_ij, g_M_ij): per-token alignment between the action
            gradient and motion gradient at the representation coordinate, or 0
            when either gradient vanishes at that token.
    w_ij  = (1 + C_ij) / (mean_valid(1 + C) + eps): normalized to mean 1 over the
            valid tokens of each sample, so L_M^AC keeps the same scale as the
            uniform mean and only redistributes credit across tokens.
    L_M^AC = (1/N_valid) * sum_ij stopgrad(w_ij) * L_ij

    Matched controls for the experiment matrix reuse the same pipeline:
    w_mode="random" replaces C with iid U(-1,1) (1 + C stays nonnegative, so the
    control can never turn the motion loss adversarial); it consumes no real
    gradients, so g_A_tok/g_M_tok may be None and compat is reported as None
    (per-token w statistics of a random draw carry no diagnostic value).
    w_mode="shuffled" permutes the learned C within the valid anchors of each
    (sample, view) (bijective, so E[w]=1 survives; the per-view valid C
    distribution is preserved, only the w_ij <-> spatial correspondence is
    broken); it still needs the real gradients, and compat is reported as the
    pre-permutation map.

    :param per_token_motion_loss: torch.Tensor, [B, V, H_tok, W_tok].
    :param g_A_tok: torch.Tensor [B, V, H_tok, W_tok, D], action-loss gradient at visual
        tokens; None for w_mode="random".
    :param g_M_tok: torch.Tensor [B, V, H_tok, W_tok, D], motion-loss gradient at visual
        tokens; None for w_mode="random".
    :param valid: torch.Tensor, [B, V, H_tok, W_tok] bool, anchors with RAFT supervision.
    :param w_mode: str, one of {"compat", "random", "shuffled"}.
    :return: (w [B, V, H_tok, W_tok] detached, compat [B, V, H_tok, W_tok] detached or None for w_mode="random").
    """
    if w_mode == "random":
        compat = None
        C_source = torch.rand_like(per_token_motion_loss) * 2.0 - 1.0
    else:
        g_A = g_A_tok.float()
        g_M = g_M_tok.float()
        compat = F.cosine_similarity(g_A, g_M, dim=-1)  # [B, V, H_tok, W_tok]
        compat = torch.where((g_A.norm(dim=-1) > 0) & (g_M.norm(dim=-1) > 0), compat, torch.zeros_like(compat))
        if w_mode == "shuffled":
            flat = compat.reshape(compat.shape[0], compat.shape[1], -1)
            vmask = valid.reshape(valid.shape[0], valid.shape[1], -1)
            C_source = torch.zeros_like(flat)
            for b in range(flat.shape[0]):
                for v in range(flat.shape[1]):
                    idx = vmask[b, v].nonzero(as_tuple=True)[0]
                    vals = flat[b, v, idx]
                    C_source[b, v, idx] = vals[torch.randperm(vals.numel(), device=vals.device)]
            C_source = C_source.view_as(compat)
        else:
            C_source = compat
    base = 1.0 + C_source
    base = torch.where(valid, base, torch.zeros_like(base))  # invalid anchors get w=0 by construction
    w = base / (base.sum(dim=(1, 2, 3), keepdim=True) / valid.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1) + 1e-6)
    return w.detach(), None if compat is None else compat.detach()

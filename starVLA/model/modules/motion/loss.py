import torch
import torch.nn.functional as F


def token_motion_loss(pred_flow: torch.Tensor, target_flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token motion loss: L_ij = 1/2 ||f_hat_ij - f*_ij||_2^2 over the 2 flow
    channels, kept per-token for spatial credit assignment.

    Anchor validity comes from the target: rows whose future frame fell outside
    the episode (no RAFT supervision) are stored as NaN and report valid=False
    with a zero loss contribution.

    :param pred_flow: torch.Tensor, [B, V, H_tok, W_tok, 2].
    :param target_flow: torch.Tensor, [B, V, H_tok, W_tok, 2]; NaN marks invalid anchors.
    :return: (per_token_loss [B, V, H_tok, W_tok], valid [B, V, H_tok, W_tok] bool).
    """
    valid = target_flow.isfinite().all(dim=-1)
    loss = 0.5 * ((pred_flow - target_flow) ** 2).sum(dim=-1)
    return torch.where(valid, loss, torch.zeros_like(loss)), valid


def motion_credit(
    per_token_motion_loss: torch.Tensor,
    g_A_tok: torch.Tensor,
    g_M_tok: torch.Tensor,
    valid: torch.Tensor,
    w_mode: str = "compat",
) -> tuple[torch.Tensor, torch.Tensor]:
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
    w_mode="random" replaces C with iid N(0,1); w_mode="shuffled" applies a
    within-sample random permutation to the learned C (bijective, so E[w]=1
    survives), breaking the w_ij <-> C_ij correspondence only.

    :param per_token_motion_loss: torch.Tensor, [B, V, H_tok, W_tok].
    :param g_A_tok: torch.Tensor, [B, V, H_tok, W_tok, D], action-loss gradient at visual tokens.
    :param g_M_tok: torch.Tensor, [B, V, H_tok, W_tok, D], motion-loss gradient at visual tokens.
    :param valid: torch.Tensor, [B, V, H_tok, W_tok] bool, anchors with RAFT supervision.
    :param w_mode: str, one of {"compat", "random", "shuffled"}.
    :return: (w [B, V, H_tok, W_tok] detached, compat [B, V, H_tok, W_tok] detached).
    """
    g_A = g_A_tok.float()
    g_M = g_M_tok.float()
    compat = F.cosine_similarity(g_A, g_M, dim=-1)  # [B, V, H_tok, W_tok]
    compat = torch.where((g_A.norm(dim=-1) > 0) & (g_M.norm(dim=-1) > 0), compat, torch.zeros_like(compat))
    if w_mode == "random":
        C_source = torch.randn_like(compat)
    elif w_mode == "shuffled":
        flat = compat.reshape(compat.shape[0], -1)
        perm = torch.argsort(torch.rand_like(flat), dim=1)
        C_source = flat.gather(1, perm).view_as(compat)
    else:
        C_source = compat
    base = 1.0 + C_source
    base = torch.where(valid, base, torch.zeros_like(base))  # invalid anchors get w=0 by construction
    w = base / (base.sum(dim=(1, 2, 3), keepdim=True) / valid.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1) + 1e-6)
    return w.detach(), compat.detach()

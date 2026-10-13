import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_to_token_grid(flow: torch.Tensor, token_hw: tuple[int, int]) -> torch.Tensor:
    """
    Project a full-resolution RAFT flow field onto the VLM visual token grid.

    :param flow: torch.Tensor, [B, 2, H, W], RAFT optical flow in pixels.
    :param token_hw: tuple[int, int], the visual token grid (H_tok, W_tok).
    :return: torch.Tensor, [B, H_tok, W_tok, 2], per-token displacement in
             token-cell units (1.0 = the motion crosses one full token cell,
             horizontal / (W / W_tok) and vertical / (H / H_tok)).
    """
    _, _, H, W = flow.shape
    H_tok, W_tok = token_hw
    # Area average pooling over each token cell: [B, 2, H_tok, W_tok]
    pooled = F.adaptive_avg_pool2d(flow, (H_tok, W_tok))
    # Per-component strides so non-square grids stay correct: x is W-strided, y is H-strided
    pooled = torch.stack([pooled[:, 0] * W_tok / W, pooled[:, 1] * H_tok / H], dim=1)
    # [B, 2, H_tok, W_tok] -> [B, H_tok, W_tok, 2]
    return pooled.permute(0, 2, 3, 1).contiguous()


def flow_confidence_q(
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    token_hw: tuple[int, int],
    q_min: float = 0.25,
    alpha1: float = 0.01,
    alpha2: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Token-cell motion confidence from loop-closure consistency (docs/research/new_idea3.md,
    supplementary spec):

    F_b^w(x) = F_{t+dt->t}(x + F_f(x))      backward flow sampled at the forward endpoint
    r(x)     = ||F_f(x) + F_b^w(x)||_2      loop-residual, pixel units
    T^2(x)   = alpha1 * (||F_f(x)||^2 + ||F_b^w(x)||^2) + alpha2
             (alpha1 = 0.01, alpha2 = 0.5: the standard forward-backward consistency
              occlusion scale, adopted as a fixed initial definition per the
              docs/research/new_idea3_review.md round-3/4 calibration)
    q_pix(x) = exp(-relu(r(x) - T(x)) / (T(x) + eps));  0 where x + F_f(x) falls outside the frame
    q_cell   = mean of q_pix over the token cell;  q = q_min + (1 - q_min) * q_cell

    Calibration note: the tight triangle-inequality bound that made alpha1 = alpha2 = 1
    in pixel units nearly uninformative (q_pix >= exp(-(sqrt(2)-1)) ~ 0.66 for any
    in-bounds pixel) no longer applies at alpha1 = 0.01, so unilateral / occluded / fused
    inconsistent flow now drives q toward the floor while exact loop closure keeps q = 1.

    :param flow_fwd: torch.Tensor, [B, 2, H, W], forward flow F_{t->t+dt} in pixels (RAFT grid resolution).
    :param flow_bwd: torch.Tensor, [B, 2, H, W], backward flow F_{t+dt->t} in the same resolution.
    :param token_hw: tuple[int, int], the visual token grid (H_tok, W_tok).
    :return: torch.Tensor, [B, H_tok, W_tok], per-token confidence in [q_min, 1].
    """
    _, _, H, W = flow_fwd.shape
    H_tok, W_tok = token_hw

    ys_base, xs_base = torch.meshgrid(
        torch.arange(H, device=flow_fwd.device), torch.arange(W, device=flow_fwd.device), indexing="ij"
    )
    base = torch.stack([xs_base, ys_base]).to(flow_fwd.dtype)
    disp = base.unsqueeze(0) + flow_fwd  # [B, 2, H, W] forward endpoint in pixels
    xs, ys = disp[:, 0], disp[:, 1]
    in_bounds = (xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)  # [B, H, W]

    grid = torch.stack(
        [xs / (W - 1) * 2 - 1, ys / (H - 1) * 2 - 1], dim=-1
    )  # [B, H, W, 2], pixel-center convention
    flow_bwd_w = F.grid_sample(flow_bwd, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    f_f = flow_fwd  # pixel units
    f_b = flow_bwd_w
    r = torch.linalg.vector_norm(f_f.add(f_b), dim=1)
    threshold = torch.sqrt(alpha1 * ((f_f * f_f).sum(1) + (f_b * f_b).sum(1)) + alpha2)
    q_pix = torch.exp(-torch.clamp(r - threshold, min=0.0) / (threshold + eps))
    q_pix = q_pix * in_bounds
    q_cell = F.adaptive_avg_pool2d(q_pix.unsqueeze(1), (H_tok, W_tok)).squeeze(1)
    return q_min + (1.0 - q_min) * q_cell


class MotionHead(nn.Module):
    """
    Pointwise readout head that decodes per-token visual representations into
    the flow field predicted at that token's spatial location.

    Deliberately a minimal pointwise MLP: it reads only the local per-token
    representation, so any motion information it can decode must already be
    present in the token representation itself.

    Language conditioning via FiLM (docs/research/new_idea3.md z_lang spec):
    z_lang is the stop-gradient pooled raw-instruction entry embedding; LN(z_lang)
    feeds a small projection whose last layer is zero-initialised, so at init
    (gamma=beta=0) the head reduces exactly to the original LayerNorm+MLP.

    :param dim: int, hidden size D of the tapped representation.
    :param hidden_dims: list of intermediate widths.
    """

    def __init__(self, dim: int, hidden_dims: tuple[int, ...] = (256, 128)):
        super().__init__()
        layers = [nn.LayerNorm(dim)]
        d = dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers)
        self.film = nn.Sequential(nn.Linear(dim, hidden_dims[0]), nn.GELU(), nn.Linear(hidden_dims[0], 2 * dim))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.z_norm = nn.LayerNorm(dim)

    def forward(self, E_vis: torch.Tensor, z_lang: torch.Tensor = None) -> torch.Tensor:
        """
        :param E_vis: torch.Tensor, [B, V, H_tok, W_tok, D] visual token representations.
        :param z_lang: torch.Tensor, [B, D] sg-pooled raw-instruction entry embeddings, or None
                       to keep the original pointwise path.
        :return: torch.Tensor, [B, V, H_tok, W_tok, 2] predicted per-token flow.
        """
        h = self.net[0](E_vis)
        if z_lang is not None:
            z_lang = z_lang.to(device=h.device, dtype=h.dtype)
            gamma, beta = self.film(self.z_norm(z_lang)).chunk(2, dim=-1)  # [B, D] each
            h = (1.0 + gamma[:, None, None, None]) * h + beta[:, None, None, None]
        return self.net[1:](h)

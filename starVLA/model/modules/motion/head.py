import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_to_token_grid(flow: torch.Tensor, token_hw: tuple[int, int]) -> torch.Tensor:
    """
    Project a full-resolution RAFT flow field onto the VLM visual token grid.

    :param flow: torch.Tensor, [B, 2, H, W], RAFT optical flow in pixels.
    :param token_hw: tuple[int, int], the visual token grid (H_tok, W_tok).
    :return: torch.Tensor, [B, H_tok, W_tok, 2], per-token displacement in
             token-cell units (1.0 = the motion crosses one full token cell).
    """
    _, _, H, W = flow.shape
    H_tok, W_tok = token_hw
    # Area average pooling over each token cell: [B, 2, H_tok, W_tok]
    pooled = F.adaptive_avg_pool2d(flow, (H_tok, W_tok))
    # Normalize by token stride so the flow unit becomes token cells
    stride = float(W) / float(W_tok)
    pooled = pooled / stride
    # [B, 2, H_tok, W_tok] -> [B, H_tok, W_tok, 2]
    return pooled.permute(0, 2, 3, 1).contiguous()


class MotionHead(nn.Module):
    """
    Pointwise readout head that decodes per-token visual representations into
    the flow field predicted at that token's spatial location.

    Deliberately a minimal pointwise MLP: it reads only the local per-token
    representation, so any motion information it can decode must already be
    present in the token representation itself.

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

    def forward(self, H_vis: torch.Tensor) -> torch.Tensor:
        """
        :param H_vis: torch.Tensor, [B, V, H_tok, W_tok, D] visual token representations.
        :return: torch.Tensor, [B, V, H_tok, W_tok, 2] predicted per-token flow.
        """
        return self.net(H_vis)

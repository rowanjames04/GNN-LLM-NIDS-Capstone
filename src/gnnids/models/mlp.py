"""Residual MLP over flow features.

Deliberately dual-purpose. In Phase 3 this is the topology-blind baseline whose
gap to the GNN measures what message passing is worth (commitment V2). In Phase
4 the same class becomes **Channel 1** of the dual-channel architecture.

Sharing one implementation is the point: if the baseline and the channel were
separate code, any measured difference between them could be an artefact of
depth, width, initialisation or optimiser rather than of topology. Here the only
difference is what the model can see.

Residual connections because depth here is about feature interaction, not
receptive field -- there is no neighbourhood to expand into -- and they keep
gradients healthy in a network that is otherwise just stacked linear layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class FlowMLP(nn.Module):
    """Encoder plus two heads: binary (headline) and multi-class (feeds the LLM).

    Two heads from a shared trunk, per D3. The binary head is the only one
    meaningful under leave-one-attack-out, since a held-out family has no valid
    multi-class target; the multi-class head supplies the attack category the
    LLM evidence pack needs, and acts as an auxiliary task that may regularise
    the binary objective -- to be confirmed by ablation rather than assumed.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        n_blocks: int = 3,
        n_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head_binary = nn.Linear(hidden_dim, 1)
        self.head_multiclass = nn.Linear(hidden_dim, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Representation before the heads — this is what Channel 1 contributes
        to the fusion in Phase 4."""
        return self.blocks(self.input_proj(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.head_binary(z).squeeze(-1), self.head_multiclass(z)


class FocalLoss(nn.Module):
    r"""Weighted focal loss (D8), replacing the proposal's SMOTE.

    .. math:: \mathcal{L} = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)

    At ~4% prevalence, plain cross-entropy is dominated by the benign majority:
    each easy negative contributes little, but there are 24 times as many of
    them, so their summed gradient swamps the attack signal. The modulating
    factor :math:`(1-p_t)^\gamma` suppresses examples the model already gets
    right -- at :math:`\gamma=2`, an example predicted at 0.9 confidence has its
    loss scaled by 0.01 -- so training attention concentrates on the hard cases,
    which in intrusion detection are the covert attacks that matter most.

    SMOTE is not usable here: it interpolates feature vectors, and a synthetic
    flow interpolated between two real ones has no valid endpoints, so it would
    fabricate topology the GNN is supposed to be learning.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        targets = targets.float()
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = -alpha_t * (1 - p_t).clamp(min=1e-8).pow(self.gamma) * p_t.clamp(min=1e-8).log()
        return loss.mean()

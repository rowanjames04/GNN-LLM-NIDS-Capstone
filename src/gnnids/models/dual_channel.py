"""The dual-channel GNN. Adapted from Bao & Wang (2024), restated at edge level.

A single GNN has a blind spot. Message passing is *good* at neighbourhood
context and, in being so, smooths away what is individual about one flow. But
some attacks are visible only in a flow's own features (a malformed packet, an
anomalous TTL) and others only in structure (fan-out, lateral movement). The
design refuses to choose:

    Channel 1  MLP over the flow's own features.      Topology-blind.
    Channel 2  Stacked E-GraphSAGE.                    Context-only.
    Fusion     Learned attention over the two, per flow.

**Channel 1 is literally the Phase 3 baseline class.** `FlowMLP` is imported
rather than reimplemented, so any measured difference between the baseline and
this model is topology and not an artefact of depth, width, initialisation or
optimiser. That is the whole point of the V2 test.

The attention weights earn their place twice. They adapt per flow -- a DoS can
lean topological while a malformed packet leans attribute -- and they *are* an
explanation, readable directly into the evidence pack at no extra cost. A flow
with high topological weight is one a flat classifier would have missed, which
makes the project's central claim visible per-detection rather than only in
aggregate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .egraphsage import TopologicalChannel
from .mlp import FlowMLP


class DualChannelGNN(nn.Module):
    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int = 128,
        n_mlp_blocks: int = 3,
        n_gnn_layers: int = 2,
        n_classes: int = 10,
        dropout: float = 0.1,
        bidirectional: bool = True,
        use_channel1: bool = True,
        use_channel2: bool = True,
    ) -> None:
        super().__init__()
        if not (use_channel1 or use_channel2):
            raise ValueError("at least one channel must be enabled")

        # Ablation switches, so "Channel 1 only" and "Channel 2 only" are the
        # same code path as the full model rather than separate scripts.
        self.use_channel1, self.use_channel2 = use_channel1, use_channel2

        if use_channel1:
            # The Phase 3 baseline, reused. Its heads are unused here; only the
            # encoder feeds the fusion.
            self.channel1 = FlowMLP(
                edge_dim, hidden_dim, n_mlp_blocks, n_classes, dropout
            )
        if use_channel2:
            self.channel2 = TopologicalChannel(
                edge_dim, hidden_dim, n_gnn_layers, dropout, bidirectional
            )

        if use_channel1 and use_channel2:
            # Attention over the two channel representations, computed per edge.
            self.fusion_score = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 2),
            )

        self.head_binary = nn.Linear(hidden_dim, 1)
        self.head_multiclass = nn.Linear(hidden_dim, n_classes)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Returns (binary logits, multi-class logits, fusion weights).

        Fusion weights are ``[n_edges, 2]`` as (attribute, topological), or None
        when only one channel is active.
        """
        z1 = self.channel1.encode(edge_attr) if self.use_channel1 else None
        z2 = self.channel2(x, edge_index, edge_attr) if self.use_channel2 else None

        if z1 is not None and z2 is not None:
            alpha = torch.softmax(self.fusion_score(torch.cat([z1, z2], dim=-1)), dim=-1)
            z = alpha[:, :1] * z1 + alpha[:, 1:] * z2
        else:
            z, alpha = (z1 if z1 is not None else z2), None

        return self.head_binary(z).squeeze(-1), self.head_multiclass(z), alpha


def channel_attribution(alpha: torch.Tensor) -> dict:
    """Summarise fusion weights over a batch of detections.

    ``topological_share`` is the aggregate answer to V2 made per-detection: the
    proportion of decision weight that came from neighbourhood context rather
    than from the flow itself. A high value on true positives means those
    detections rest on information a flat classifier does not have.
    """
    if alpha is None:
        return {}
    alpha = alpha.detach()
    return {
        "attribute_weight_mean": round(float(alpha[:, 0].mean()), 4),
        "topological_weight_mean": round(float(alpha[:, 1].mean()), 4),
        "topological_share_above_half": round(float((alpha[:, 1] > 0.5).float().mean()), 4),
    }

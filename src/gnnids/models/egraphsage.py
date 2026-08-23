"""E-GraphSAGE message passing: aggregate EDGE features into node embeddings.

PyTorch Geometric ships no layer for this, because the usual assumption is that
nodes carry the features. Network flow data is the other way round: an IP
address has no intrinsic properties, while every measurement -- bytes, duration,
TCP flags, packet-size distribution -- lives on the flow. Stock `SAGEConv` would
discard the entire feature set.

The update rule (Lo et al., 2022):

.. math::
    \\mathbf{h}_{\\mathcal{N}(v)}^{(k)} = \\mathrm{AGG}\\left(\\left\\{
        \\mathbf{h}_u^{(k-1)} \\,\\Vert\\, \\mathbf{x}_{uv} :
        (u,v) \\in \\mathcal{N}(v) \\right\\}\\right)

.. math::
    \\mathbf{h}_v^{(k)} = \\sigma\\!\\left(\\mathbf{W}^{(k)} \\cdot \\left[
        \\mathbf{h}_v^{(k-1)} \\,\\Vert\\, \\mathbf{h}_{\\mathcal{N}(v)}^{(k)}
    \\right]\\right)

Nodes are initialised to a constant, so **a host's representation is entirely a
function of the traffic passing through it**. That is what makes the model
inductive: embeddings are computed from features and neighbourhood rather than
looked up per address, so unseen IPs work without retraining -- a precondition
for both leave-one-attack-out and cross-dataset transfer.

Aggregation must be permutation-invariant: a host's connections have no
canonical order, and a model sensitive to that order would be learning an
artefact of how the data happens to be stored.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EGraphSAGEConv(MessagePassing):
    """One round of edge-feature message passing.

    Directed by default (D21-adjacent reasoning: "A scanned B" and "B scanned A"
    are different incidents, and the LLM evidence pack needs the distinction).
    Setting ``bidirectional`` also passes messages backwards along each edge,
    which lets a victim's embedding reflect who contacted it -- useful for
    fan-in patterns like DoS.
    """

    def __init__(
        self,
        in_node_dim: int,
        edge_dim: int,
        out_dim: int,
        aggr: str = "mean",
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__(aggr=aggr, node_dim=0)
        self.bidirectional = bidirectional

        # The message is [neighbour embedding || edge features], projected down
        # before aggregation so the edge features do not dominate by width.
        self.msg_proj = nn.Sequential(
            nn.Linear(in_node_dim + edge_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Self-representation concatenated with the aggregate, per GraphSAGE --
        # a node keeps its own state rather than being overwritten by its
        # neighbourhood, which is what limits over-smoothing across layers.
        self.update_proj = nn.Sequential(
            nn.Linear(in_node_dim + out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        aggregated = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        if self.bidirectional:
            reverse = torch.stack([edge_index[1], edge_index[0]])
            aggregated = aggregated + self.propagate(
                reverse, x=x, edge_attr=edge_attr
            )

        return self.update_proj(torch.cat([x, aggregated], dim=-1))

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """x_j is the source node of each edge; edge_attr is that flow."""
        return self.msg_proj(torch.cat([x_j, edge_attr], dim=-1))


class TopologicalChannel(nn.Module):
    """Channel 2: stacked E-GraphSAGE layers, then an edge representation.

    ``K`` should be at least the number of hops an attack propagates over --
    lateral movement spans several, and a single layer cannot represent it. But
    depth is limited by over-smoothing, where embeddings converge and become
    indistinguishable, so 2-3 is the usual useful range.
    """

    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = 1  # constant node initialisation
        for _ in range(n_layers):
            self.layers.append(
                EGraphSAGEConv(in_dim, edge_dim, hidden_dim, dropout=dropout,
                               bidirectional=bidirectional)
            )
            in_dim = hidden_dim

        # An edge is represented by both its endpoints. Source and destination
        # are kept in order rather than summed, so the aggressor/victim
        # direction survives into the classifier and the evidence pack.
        self.edge_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        src, dst = edge_index
        return self.edge_proj(torch.cat([x[src], x[dst]], dim=-1))

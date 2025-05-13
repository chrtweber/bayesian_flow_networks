import torch
import torch.nn as nn

import numpy as np

from torch_bfn.networks.base import BFNetwork


class NeuralSCM(BFNetwork):
    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        adj_matrix: np.ndarray,
        topo_order: list,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.adj_matrix = adj_matrix
        self.topo_order = topo_order
        self.f = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        int(adj_matrix[i].sum()) + 1, hidden_dim
                    ),  # +1 for noise uᵢ
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for i in range(num_vars)
            ]
        )

    def forward(self, u: torch.Tensor, interventions: dict) -> torch.Tensor:
        """
        Args:
            u: noise vector [B, D]
            interventions: dict of {var_idx: fixed value}

        Returns:
            z: latent causal variables [B, D]
        """
        B, D = u.shape
        z = torch.zeros_like(u)

        for i in self.topo_order:
            if i in interventions:
                z[:, i] = interventions[i]
            else:
                pa_idx = np.where(self.adj_matrix[i] == 1)[0].tolist()
                if pa_idx:
                    inputs = torch.cat([z[:, pa_idx], u[:, i : i + 1]], dim=1)
                else:
                    inputs = u[:, i : i + 1]
                z[:, i] = self.f[i](inputs).squeeze(1)
        return z

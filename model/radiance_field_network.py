from math import exp, log

import torch
import torch.nn as nn


class RadianceFieldNetwork(nn.Module):
    def __init__(
        self,
        layers=5,
        hidden_dim=64,
        encoding=True,
        output='clamped',
        psi=4,
        levels=16,
        feature_dim=2,
        pos_fine_res=512,
        dir_fine_res=128,
        pos_coarse_res=16,
        dir_coarse_res=16,
        hash_table_size=14,
    ):
        super().__init__()

        if encoding:
            self.position_encoding = MultiresolutionHashEncoding(
                3,
                hash_table_size,
                pos_fine_res,
                levels=levels,
                feature_dim=feature_dim,
                coarse_res=pos_coarse_res,
            )
            self.direction_encoding = MultiresolutionHashEncoding(
                2,
                hash_table_size,
                dir_fine_res,
                levels=levels,
                feature_dim=feature_dim,
                coarse_res=dir_coarse_res,
            )
            pos_size = self.position_encoding.L * self.position_encoding.F
            dir_size = self.direction_encoding.L * self.direction_encoding.F
        else:
            self.position_encoding = nn.Identity()
            self.direction_encoding = nn.Identity()
            pos_size = 3
            dir_size = 2

        assert layers >= 1, 'Number of layers must be at least 1.'

        self.fc = nn.ModuleList(
            [
                nn.Linear(
                    pos_size + dir_size if i == 0 else hidden_dim,
                    3 if i == layers - 1 else hidden_dim,
                )
                for i in range(layers)
            ]
        )

        assert output in ['clamped', 'sigmoid', 'log_radiance'], (
            'Output type must be "clamped", "sigmoid", or "log_radiance".'
        )
        self.output = output
        self.psi = psi

    def forward(self, x):
        pos = self.position_encoding(x[:, :3])
        dir = self.direction_encoding(x[:, 3:])
        x = torch.cat([pos, dir], dim=-1)
        for layer in self.fc:
            x = torch.relu(layer(x))
        if self.output == 'clamped':
            x = torch.clamp(x, min=0.0, max=1.0)
        elif self.output == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.output == 'log_radiance':
            x = torch.pow(10, -x * self.psi)
        return x


class MultiresolutionHashEncoding(nn.Module):
    """Multiresolution Hash Encoding from the paper "Instant Neural Graphics
    Primitives with a Multiresolution Hash Encoding" by Müller et al. (2022).

    Parameters:
        - input_dim: 2 for angles, 3 for positions.             ... d
        - hash_table_size: Maximum size of each hash table.     ... T
        - fine_res: Resolution of the finest level.             ... N_max
        - levels: Number of levels in the hierarchy.            ... L
        - feature_dim: Number of features per level.            ... F
        - coarse_res: Resolution of the coarsest level.         ... N_min

    According to the paper, only the hash table size and fine resolution
    need to be tuned to the task.

    Paper: https://arxiv.org/pdf/2201.05989
    """

    def __init__(
        self,
        input_dim,
        hash_table_size=14,
        fine_res=512,
        *,
        levels=16,
        feature_dim=2,
        coarse_res=16,
    ):
        super().__init__()
        self.L = levels
        self.T = 2**hash_table_size
        self.F = feature_dim
        self.N_min = coarse_res
        self.N_max = fine_res

        assert input_dim in [2, 3], 'Input dimension must be either 2 or 3.'
        self.d = input_dim

        self.b = exp((log(self.N_max) - log(self.N_min)) / (self.L - 1))
        self.N_l = [int(self.N_min * (self.b**level)) for level in range(self.L)]

        self.tables = nn.ModuleList(
            [nn.Embedding(self._table_size(level), self.F) for level in range(self.L)]
        )

        # Initialize using He initialization
        for table in self.tables:
            nn.init.kaiming_normal_(table.weight)

    def _table_size(self, level):
        # Embedding tables for coarser levels can be smaller
        return min((self.N_l[level] + 1) ** self.d, self.T)

    def _spatial_hash(self, x):
        primes = torch.tensor(
            [1, 2654435761, 805459861], device=x.device, dtype=x.dtype
        )[: self.d]

        hashed = torch.bitwise_xor((x * primes)[..., 0], (x * primes)[..., 1])
        if self.d == 3:
            hashed = torch.bitwise_xor(hashed, (x * primes)[..., 2])

        return hashed % self.T

    def _one_one_mapping(self, x, N_l):
        res = N_l + 1
        powers = res ** torch.arange(self.d, device=x.device).flip(0)
        return (x * powers).sum(dim=-1)

    def _hash_fn(self, corners, N_l, one_one_mapping):
        return (
            self._one_one_mapping(corners, N_l)
            if one_one_mapping
            else self._spatial_hash(corners)
        )

    def forward(self, x):
        features = []

        for N_l, table in zip(self.N_l, self.tables):
            # Compute top left (or front) corner of the voxel containing x
            # shape: (batch_size, d)
            voxel_corner = torch.floor(x * N_l)

            # Create all 4 or 8 corner indices
            # shape: (batch_size, 2^d, d)
            offsets = torch.cartesian_prod(
                *[torch.tensor([0, 1], device=x.device)] * self.d
            ).unsqueeze(0)
            corners = voxel_corner.unsqueeze(1) + offsets

            one_one = (N_l + 1) ** self.d <= self.T

            # 1. Hashing (finer levels) of 1:1 mapping (coarser levels)
            # shape: (batch_size, 2^d)
            indices = self._hash_fn(corners.long(), N_l, one_one)

            # 2. Lookup
            # shape: (batch_size, 2^d, F)
            embeddings = table(indices)

            # 3. Linear interpolation w_l := x_l - floor(x_l)
            # shape: (batch_size, 2^d)
            w_l = (x * N_l - voxel_corner).unsqueeze(1)
            corner_weights = offsets * w_l + (1 - offsets) * (1 - w_l)
            w = corner_weights.prod(dim=-1)

            # shape: (batch_size, F)
            weighted = (embeddings * w.unsqueeze(-1)).sum(dim=1)
            features.append(weighted)

        # 4. Concatenation of all levels
        return torch.cat(features, dim=-1)

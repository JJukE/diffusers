""" Utils and modules for points """
import numpy as np
import torch
from torch import nn


def normalize_3d_coordinate(xyz: torch.Tensor, padding: float):
    # xyz_norm = xyz / (1 + padding + 1e-3)  # [-1, 1] -> [-0.9, 0.9]
    xyz_norm = xyz * (1 - padding)  # [-1, 1], padding=0.1 -> [-0.9, 0.9]
    # xyz_norm = (xyz_norm + 1) / 2  # [-0.9, 0.9] -> [0.1, 1.9] -> [0.05, 0.95]
    return xyz_norm


class RFFPointEmbedding(nn.Module):
    """ Random Fourier features per-point embedding from https://arxiv.org/bas/2309.00339 """
    
    def __init__(self, out_channels, scale=0.9, pooling="mean"):
        super().__init__()
        
        assert pooling in ["max", "mean", "median"]
        self.scale = scale # 0.09 for max pooling and 0.9 for mean or median pooling
        self.pooling = pooling
        self.out_channels = out_channels
        
    
    def forward(self, input_pts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_pts (B, N, c): Input point cloud
        Returns:
            embeddings (B, N, C): Point embeddings
        """
        b = self.scale * torch.randn((int(self.out_channels / 2), input_pts.shape[2])).cuda() # (C/2, c)
        return torch.cat((torch.sin((2. * np.pi * input_pts) @ b.T), torch.cos((2. * np.pi * input_pts) @ b.T)), -1)


class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        h_dim (int): hidden dimension
    """

    def __init__(self, in_dim, out_dim=None, h_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim

        if h_dim is None:
            h_dim = min(in_dim, out_dim)

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        
        self.fc_0 = nn.Linear(in_dim, h_dim)
        self.fc_1 = nn.Linear(h_dim, out_dim)
        self.nonlinearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)  # nn.ReLU()

        if in_dim == out_dim:
            self.use_shortcut = None
        else:
            self.use_shortcut = nn.Linear(in_dim, out_dim, bias=False)
        
        # zero out the params
        nn.init.zeros_(self.fc_1.weight)


    def forward(self, input_tensor: torch.FloatTensor):
        hidden_states = self.fc_0(self.nonlinearity(input_tensor))
        hidden_states = self.fc_1(self.nonlinearity(hidden_states))

        input_tensor = self.use_shortcut(input_tensor) if self.use_shortcut is not None else input_tensor

        return input_tensor + hidden_states
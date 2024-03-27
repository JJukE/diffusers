import numpy as np
import torch
from torch import nn


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
        self.nonlinearity = nn.LeakyReLU(neg_slope=0.1, inplace=True)  # nn.ReLU()

        if in_dim == out_dim:
            self.use_shortcut = None
        else:
            self.use_shortcut = nn.Linear(in_dim, out_dim, bias=False)
        
        # zero out the params
        nn.init.zeros_(self.fc_1.weight)


    def forward(self, input_tensor: torch.FloatTensor):
        hidden_states = self.fc_0(self.nonlinearity(input_tensor))
        hidden_states = self.fc_1(self.nonlinearity(hidden_states))

        input_tensor = self.use_shortcut(input_tensor) if self.use_shorcut is not None else input_tensor

        return input_tensor + hidden_states


class TriplaneConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, is_rollout=True) -> None:
        """ 3D aware convolution for triplane features
        from https://github.com/Sin3DM/Sin3DM
        """
        super().__init__()
        
        in_channels = in_channels * 3 if is_rollout else in_channels
        self.is_rollout = is_rollout
        
        self.conv_xy = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv_xz = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv_yz = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    
    def forward(self, feat_planes, scale=None):
        """
        Args:
            feat_planes: (3B, C, H, W)
        Returns:
            feat_planes: (3B, C_out, H, W)
        """
        # (B, C, res_x + res_z, res_y + res_z) == (B, C, H+D, W+D)
        xy, xz, yz = feat_planes.chunk(3, dim=0)
        res_x, res_y = xy.shape[-2:]
        res_z = xz.shape[-1]
        
        if self.is_rollout:
            xy_hidden = torch.cat([xy, # (B, C, res_x, res_y)
                torch.mean(yz, dim=-1, keepdim=True).transpose(-1, -2).expand_as(xy), # (B, C, res_y, 1) -> (B, C, 1, res_y) -> (B, C, res_x, res_y)
                torch.mean(xz, dim=-1, keepdim=True).expand_as(xy) # (B, C, res_x, 1) -> (B, C, res_x, res_y)
            ], dim=1) # (B, 3C, res_x, res_y) == (B, 3C, H, W)
            xz_hidden = torch.cat([xz, # (B, C, res_x, res_z)
                torch.mean(xy, dim=-1, keepdim=True).expand_as(xz), # (B, C, res_x, 1) -> (B, C, res_x, res_z)
                torch.mean(yz, dim=-2, keepdim=True).expand_as(xz) # (B, C, 1, res_z) -> (B, C, res_x, res_z)
            ], dim=1) # (B, 3C, res_x, res_z) == (B, 3C, H, D)
            yz_hidden = torch.cat([yz, # (B, C, res_y, res_z)
                torch.mean(xy, dim=-2, keepdim=True).transpose(-1, -2).expand_as(yz), # (B, C, 1, res_y) -> (B, C, res_y, 1) -> (B, C, res_y, res_z)
                torch.mean(xz, dim=-2, keepdim=True).expand_as(yz) # (B, C, 1, res_z) -> (B, C, res_y, res_z) 
            ], dim=1) # (B, 3C, res_y, res_z) == (B, 3C, W, D)
        else:
            xy_hidden = xy
            xz_hidden = xz
            yz_hidden = yz
        
        assert xy_hidden.shape[-2] == res_x and xy_hidden.shape[-1] == res_y
        assert xz_hidden.shape[-2] == res_x and xz_hidden.shape[-1] == res_z
        assert yz_hidden.shape[-2] == res_y and yz_hidden.shape[-1] == res_z
        
        xy_hidden = self.conv_xy(xy_hidden) # 3C or C -> C_out
        xz_hidden = self.conv_xz(xz_hidden) # 3C or C -> C_out
        yz_hidden = self.conv_yz(yz_hidden) # 3C or C -> C_out
        return torch.cat([xy_hidden, xz_hidden, yz_hidden], dim=0) # (3B, C_out, H, W)
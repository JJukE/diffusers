import torch
from torch import nn


class TriplaneGroupNorm(nn.Module):
    r"""
    GroupNorm layer modified to deal with triplanes.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
        num_groups (`int`): The number of groups to separate the channels into.
        act_fn (`str`, *optional*, defaults to `None`): The activation function to use.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
    """

    def __init__(
        self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True, is_rollout: bool = True
    ):
        super().__init__()
        in_channels = num_channels * 3 if is_rollout else num_channels
        
        self.norm_xy = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
        self.norm_xz = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
        self.norm_yz = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)


    def forward(self, feat_planes: torch.Tensor) -> torch.Tensor:
        # triplane shape: (3B, C, H, W), (H = W = D)
        xy, xz, yz = feat_planes.chunk(3, dim=0)
        H, W = xy.shape[-2:]
        D = xz.shape[-1]
        
        xy_hidden = self.norm_xy(xy) # (B, C, H, W)
        xz_hidden = self.norm_xz(xz) # (B, C, H, D)
        yz_hidden = self.norm_yz(yz) # (B, C, W, D)
        
        assert xy_hidden.shape[-2] == H and xy_hidden.shape[-1] == W
        assert xz_hidden.shape[-2] == H and xz_hidden.shape[-1] == D
        assert yz_hidden.shape[-2] == W and yz_hidden.shape[-1] == D
        
        return torch.cat([xy_hidden, xz_hidden, yz_hidden], dim=0) # (3B, C_out, H, W)


class TriplaneSiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.silu = nn.SiLU()

    def forward(self, feat_planes):
        # triplane shape: (3B, C, H, W), (H = W = D)
        xy, xz, yz = feat_planes.chunk(3, dim=0)
        return torch.cat([self.silu(xy), self.silu(xz), self.silu(yz)], dim=0) # (3B, C, H, W)


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


class TriplaneConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True, is_rollout=True) -> None:
        """ 3D aware transposed convolution for triplane features """
        super().__init__()
        
        in_channels = in_channels * 3 if is_rollout else in_channels
        self.is_rollout = is_rollout
        
        self.conv_xy = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                          padding=padding, output_padding=output_padding, bias=bias, padding_mode="zeros") # "replicate"
        self.conv_xz = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                          padding=padding,output_padding=output_padding, bias=bias, padding_mode="zeros") # "replicate"
        self.conv_yz = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                          padding=padding, output_padding=output_padding, bias=bias, padding_mode="zeros") # "replicate"
    
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
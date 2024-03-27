""" based on diffusers """
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from diffusers import UNet2DModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from .scatter_ops import scatter_mean, scatter_max
from .point_utils import normalize_3d_coordinate
from .triplane_modules import RFFPointEmbedding, ResnetBlockFC, TriplaneConv


grid_sample = partial(F.grid_sample, padding_mode="border", align_corners=True)


class Downsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        downsample_steps: int = 1,
        limit_receptive_field: bool = True
    ):
        super().__init__()
        
        kernel_sizes = (2, 1) if limit_receptive_field else (3, 3)
        paddings = (0, 0) if limit_receptive_field else (1, 1)
        
        input_channel = in_channels
        
        self.blocks = []
        for _ in range(downsample_steps): # for i, down_block_type in enumerate(down_block_types):
            output_channel = input_channel * 2
            self.blocks.append(
                nn.Sequential(
                    TriplaneConv(input_channel, output_channel, kernel_size=kernel_sizes[0], stride=2, padding=paddings[0]),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    nn.GroupNorm(32, output_channel),
                    TriplaneConv(input_channel, output_channel, kernel_size=kernel_sizes[1], stride=1, padding=paddings[1]),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    nn.GroupNorm(32, output_channel),
                )
            )
            input_channel = output_channel
        self.blocks = nn.Sequential(*self.blocks)
            
    
    def forward(self, input_tensor: torch.FloatTensor) -> torch.FloatTensor:
        output_tensor = self.blocks(input_tensor)
        return output_tensor


class Upsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        upsample_steps: int = 1
    ):
        super().__init__()
        
        input_channel = in_channels
        
        self.blocks = []
        for _ in range(upsample_steps):
            output_channel = input_channel // 2
            self.blocks.append(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.TriplaneConv(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.GroupNorm(32, output_channel),
                nn.TriplaneConv(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.GroupNorm(32, output_channel)
            )
            input_channel = output_channel
        self.blocks = nn.Sequential(*self.blocks)
    
    
    def forwrad(self, input_tensor: torch.FloatTensor) -> torch.FloatTensor:
        output_tensor = self.blocks(input_tensor)
        return output_tensor

            
class PointEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_blocks: int = 5,
        triplane_res: int = 64,
        padding: float = 0.1
    ):
        super().__init__()
        
        self.triplane_res = triplane_res
        self.padding = padding
        
        self.point_emb = RFFPointEmbedding(2*channels, scale=0.9, pooling="mean")
        self.resnet_blocks = nn.ModuleList(ResnetBlockFC(2*channels, channels) for _ in range(n_blocks))
        self.proj = nn.Linear(channels, out_channels)
        self.nonlinearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    
    @torch.no_grad()
    def coordinate_to_index(self, xyz: torch.Tensor, reso: int, rounding=False) -> List[torch.Tensor]:
        """
        - xyz: (B, N, 3) [-1, 1]
        - return:
            indices: 3 x (B, 1, N), long
        """
        # xyz = ((xyz * 2 - 1) * reso).clamp_(0, reso - 1).long()
        xyz = xyz.add(1).mul_(reso / 2).clamp_(0, reso - 1)
        xyz = xyz.round().long() if rounding else xyz.long() # torch.long() = torch.int64
        indices = []

        for axis1, axis2 in [(0, 1), (0, 2), (1, 2)]:
            index = xyz[..., axis1] + reso * xyz[..., axis2]
            index.unsqueeze_(1)  # (B, 1, N)
            indices.append(index)

        return indices


    def sample_triplane_features(self, indices, feat, reso, aggregator="mean"):
        """
        Args:
            indices: list of (B, 1, N), long
            feat: (B, N, C)
        
        Returns:
            out_list: list of (B, C, H, W), float
        """
        feat = feat.transpose(1, 2).contiguous() # (B, C, N)

        out_list = []
        for index in indices:
            if aggregator == "mean":
                out = scatter_mean(feat, index, dim_size=reso**2)  # (B, C, R^2)
            elif aggregator == "max":
                out = scatter_max(feat, index, dim_size=reso**2)[0]  # (B, C, R^2)
            else:
                raise NotImplementedError(aggregator)

            out = out.view(*out.shape[:2], reso, reso).contiguous()
            out_list.append(out)

        return out_list
    
    
    def pool_local(self, indices, feat):
        """
        Args:
            indices: list of (B, 1, N), long
            feat: (B, N, C)
        
        Returns:
            pooled: (B, N, C)
        """
        # H = W = res
        feat_planes = self.sample_triplane_features(indices, feat, self.triplane_res, aggregator="mean") # (B, C, H, W)
        outs = []
        for feat_plane, index in zip(feat_planes, indices):
            feat_plane = feat_plane.flatten(2).contiguous() # (B, C, res^2)
            pooled = feat_plane.gather(2, index.expand(-1, feat_plane.shape[1], -1))
            outs.append(pooled)
        return sum(outs).transpose(1, 2).contiguous() # (B, N, C)

    
    def forward(self, input_pts: torch.Tensor):
        """
        Args:
            input_pts: (B, N, 3)
        
        Returns:
            feat_planes: (3B, C, H, W)
            feat: (B, N, C)
        """
        pts_norm = normalize_3d_coordinate(input_pts[..., :3], padding=self.padding)
        indices = self.coordinate_to_index(pts_norm, self.triplane_res)
        
        feat = self.point_emb(input_pts)
        
        feat = self.blocks[0](feat)
        for block in self.blocks[1:]:
            pooled = self.pool_local(indices, feat)
            feat = block(torch.cat([feat, pooled], dim=2))
        
        feat = self.proj(feat) # (B, N, C)
        
        feat_planes = self.sample_triplane_features(indices, feat, self.triplane_res) # (B, C, H, W)
        return torch.cat(feat_planes, dim=0), feat


class TriplaneDecoder(nn.Module):
    """ Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        feat_dim (int): dimension of latent conditioned code c (channels of triplane)
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim_out: int = 1,
        channels: int = 128,
        hidden_size: int = 256,
        n_blocks: int = 5,
        # plane_aggregation: str = "sum",
        padding: float = 0.1,
    ):
        super().__init__()
        
        self.channels = channels
        self.n_blocks = n_blocks
        self.padding = padding

        self.proj_c = nn.ModuleList([nn.Linear(channels, hidden_size) for i in range(n_blocks)])
        self.point_emb = RFFPointEmbedding(hidden_size, scale=0.9, pooling="mean") # TODO: use PE or not? # self.proj_p = nn.Linear(3, hidden_size)
        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])
        self.proj_out = nn.Linear(hidden_size, dim_out)
        self.nonlinearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.sampler = TriplaneSampler(channels=channels, dim=hidden_size, plane_aggregation=plane_aggregation, padding=padding)
    

    def sample_features_from_triplane(self, feat_planes: torch.Tensor, query_pts: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            feat_planes: (3B, C, H, W)
            query_pts: (B, N, 3)

        Returns:
            out_feat: (B, N, C)
        """
        out_feats = []
        feat_planes = feat_planes.chunk(3, dim=0) # list of (B, C, R, R)
        for feat_plane, axis1, axis2 in zip(feat_planes, [0, 0, 1], [1, 2, 2]):
            out_feat = grid_sample(feat_plane, query_pts[:, :, None, [axis1, axis2]])  # (B, C, N, 1)
            out_feat = out_feat[..., 0].transpose(1, 2).contiguous() # (B, N, C)
            out_feats.append(out_feat)
        return out_feats


    def forward(self, feat_planes, query_pts):
        feat = self.point_emb(query_pts)

        query_pts_norm = normalize_3d_coordinate(query_pts, padding=self.padding)
        plane_feats = self.sample_features_from_triplane(feat_planes, query_pts_norm)  # 3 x (B, N, C)
        plane_feat = sum(plane_feats) # TriplaneSampler (sum)

        for i in range(self.n_blocks):
            if self.channels != 0:
                feat = feat + self.proj_c[i](plane_feat)
            feat = self.blocks[i](feat)
        out = self.proj_out(self.nonlinearity(feat))
        return out


# TODO: TriplaneVAE
class TriplaneVAE(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        latent_res: int = 64,
        
        # point encoder
        pt_enc_channels: int,
        pt_channels: int,
        triplane_res: int = 128,
        padding: float = 0.1, # conventional padding parameter of OccNet for unit cube
        
        # downsampler
        downsample_steps: int = 1,
        
        # unet
        in_channels: int = 3,
        out_channels: int = 3,
        
        # upsampler
        upsample_steps: int = 1,
        
        # decoder
        dim_out: int = 1,
        channels: int = 128,
        hidden_size: int = 256,
        n_blocks: int = 5,
        # plane_aggregation: str = "sum",
        
        kl_weight: float = 1e-6,
        scaling_factor: float = None,
    ):
        super().__init__()
        
        self.kl_weight = kl_weight
        self._xyz_prove_list = None
        self.scaling_factor = scaling_factor
        
        self.point_encoder = PointEncoder(
            channels=pt_enc_channels,
            out_channels=pt_channels,
            n_blocks=5,
            triplane_res=triplane_res,
            padding=padding
        )
        
        self.downsampler = Downsampler(
            in_channels=pt_channels,
            downsample_steps=downsample_steps,
            limit_receptive_field=True
        )
        
        inter_channel = pt_channels * (2**downsample_steps)
        if kl_weight > 0:
            self.channel_encoder = nn.Sequential(
                nn.Conv2d(inter_channel, 2*latent_channels, 1),
                nn.Conv2d(2*latent_channels, 2 * latent_channels, 1) # quant_conv
            )
        else:
            self.channel_encoder = nn.Sequential(
                nn.Conv2d(inter_channel, latent_channels, 1),
                nn.Conv2d(latent_channels, latent_channels, 1) # quant_conv
            )

        # self.channel_decoder = nn.Sequential(
        #     nn.Conv2d(latent_channels, latent_channels, 1), # post_quant_conv
        #     nn.Conv2d(latent_channels, 3*inter_channel, 1)
        # )
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        
        self.unet = TriplaneUNet(
            sample_size=latent_res,
            in_channels=latent_channels,
            out_channels=3*inter_channel
            time_embedding_type=
        )
        
        self.upsampler = Upsampler(
            in_channels=inter_channel,
            upsample_steps=upsample_steps
        )
        
        self.decoder = TriplaneDecoder(
            dim_out=dim_out,
            channels=channels,
            hidden_size=hidden_size,
            n_blocks=n_blocks,
            padding=padding
        )
        
        
    def encode(self, input_pts: torch.Tensor, deterministic: bool = False):
        feat_planes, feat = self.point_encoder(input_pts) # (3B, C, H, W), (B, N, C)
        triplane = feat_planes.clone() # (3B, C, H, W)
        
        bs, ch, height, width = feat_planes.shape
        assert bs % 3 == 0
        
        feat_planes = self.downsampler(feat_planes) # (3B, C_inter, h, w)
        # feat_planes = self.unet(feat_planes)
        feat_planes = feat_planes.reshape(bs//3, 3*ch, height, width) # (B, 3*C_inter, h, w)
        feat_planes = self.channel_encoder(feat_planes) # (B, 2*latent_dim, h, w)
        
        if self.kl_weight > 0:
            posterior = DiagonalGaussianDistribution(feat_planes, deterministic=deterministic)
        else:
            var = torch.zeros_like(feat_planes)
            feat_planes = torch.cat([feat_planes, var], dim=1)
            posterior = DiagonalGaussianDistribution(feat_planes, deterministic=True)
        
        return posterior, triplane

    
    def decode1(self, z: torch.Tensor):
        # z: (B, latent_channels, h, w)
        z = z / self.scaling_factor if self.scaling_factor is not None else z # for jjuke_diffusion
        z = self.unet(self.post_quant_conv(z)) # (B, 3*C_inter, h, w)
        assert z.shape[1] % 3 == 0
        
        z = z.reshape(z.shape[0] * 3, z.shape[1] // 3, z.shape[2], z.shape[3]) # (3B, C_inter, h, w)
        return self.upsampler(z) # (3B, C, H, W)
    
    
    def decode2(self, feat_planes: torch.Tensor, query_pts: torch.Tensor):
        return self.decoder(feat_planes, query_pts)
    
    
    def forward(self, input_pts: torch.Tensor, query_pts: torch.Tensor):
        """
        Args:
            input_pts: Input pointset (B, N, 3)
            query_pts: Input query pointset (B, M, 3)
        
        Returns:
            occ: Occupancy values of the query points (B, M, 1)
            posterior: Posterior distribution
            z: Latent feature planes (B, c, h, w)
            enc_triplane: Triplane before encoding (3B, C, H, W)
            dec_triplane: Triplane after decoding (3B, C, H, W)
        """
        posterior, enc_triplane = self.encode(input_pts)
        z = posterior.sample() # (B, c, h, w)
        
        feat_planes = self.decode1(z)
        dec_triplane = feat_planes.clone() # (3B, C, H, W)
        
        occ = self.decode2(feat_planes, query_pts)
        return occ, posterior, enc_triplane, dec_triplane
    
    
    @torch.no_grad()
    def sample_from_latent(self, feat_planes: torch.Tensor, grid_res: int, padding=0.):
        """
        Args:
            feat_planes: 
        """


def __test__():
    import yaml
    from jjuke.net_utils import instantiate_from_config
    
    config_1 = """
    target: triplane_modules.ResnetBlockFC
    params:
        size_in: 64
        size_out: 32
        size_h: Null
    """
    config_1 = yaml.safe_load(config_1)
    resblock_1 = instantiate_from_config(config_1)
    
    config_2 = """
    target: diffusers.ResnetBlockCondNorm2D
    params:
        size_in: 32
        size_out: Null
        size_h: Null
    """
    config_2 = yaml.safe_load(config_2)
    resblock_2 = instantiate_from_config(config_2)
    
    x_1 = torch.rand((8, 2048, 3)) * 2 - 1 # [-1, 1]
    res_1 = resblock_1(x_1)
    
    


if __name__ == "__main__":
    __test__()
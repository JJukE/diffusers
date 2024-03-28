""" based on diffusers """
from functools import partial
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

# from .scatter_ops import scatter_mean, scatter_max
# from .point_modules import normalize_3d_coordinate, RFFPointEmbedding, ResnetBlockFC
# from triplane_modules import TriplaneConv
# from .triplane_unet_blocks import ResnetBlockTriplane, DownsampleTriplane, UpsampleTriplane
# from .triplane_unet import UNetTriplaneModel
from scatter_ops import scatter_mean, scatter_max
from point_modules import normalize_3d_coordinate, RFFPointEmbedding, ResnetBlockFC
from triplane_modules import TriplaneConv
from triplane_unet_blocks import ResnetBlockTriplane, DownsampleTriplane, UpsampleTriplane
from triplane_unet import UNetTriplaneModel


grid_sample = partial(F.grid_sample, padding_mode="border", align_corners=True)


class DownSampler(nn.Module):
    """ Triplane Channel Mixer
    Concats the triplanes along channels instead of batches. Also, downsample the triplanes for
    converting them into proper shape of latents.
    """
    def __init__(self, channels: int, in_channels: int, out_channels: int, num_sample_steps: int):
        super().__init__()

        self.conv_in = TriplaneConv(in_channels, channels, kernel_size=1, stride=1)
        
        input_channel = channels
        
        resnets, downsamplers = [], []
        for _ in range(num_sample_steps):
            output_channel = input_channel * 2
            norm_groups = 32 if input_channel > 32 else input_channel
            resnets.append(
                ResnetBlockTriplane(in_channels=input_channel, out_channels=output_channel, groups=norm_groups)
            )
            downsamplers.append( # padding=0?
                DownsampleTriplane(channels=output_channel, out_channels=output_channel, padding=1)
            )
            input_channel = output_channel
        
        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = nn.ModuleList(downsamplers)

        self.channel_encoder = nn.Sequential(
            nn.Conv2d(3*input_channel, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, 1) # quant_conv
        )
    
    
    def forward(self, feat_planes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_planes: (3B, C_in, H_in, W_in)
        Returns:
            feat_planes: (B, C_out, H_out, W_out)
        """
        feat_planes = self.conv_in(feat_planes)
        
        for resnet, sampler in zip(self.resnets, self.downsamplers):
            feat_planes = resnet(feat_planes, temb=None)
            feat_planes = sampler(feat_planes)
        
        xy_hidden, xz_hidden, yz_hidden = feat_planes.chunk(3, dim=0)
        feat_planes = torch.cat([xy_hidden, xz_hidden, yz_hidden], dim=1) # (B, 3C, H, W)
        
        return self.channel_encoder(feat_planes)


class UpSampler(nn.Module):
    """ UpSampler for triplanes
    Decodes the latents and concat along batches instead of channels.
    Also, upsample the latents for converting them into proper shape of triplanes.
    """
    def __init__(self, channels: int, in_channels: int, out_channels: int, num_sample_steps: int):
        super().__init__()
        self.channel_decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), # post_quant_conv
            nn.Conv2d(in_channels, 3*channels, 1)
        )
        
        input_channel = channels
        
        resnets, upsamplers = [], []
        for _ in range(num_sample_steps):
            output_channel = input_channel // 2
            norm_groups = 32 if output_channel > 32 else output_channel
            resnets.append(
                ResnetBlockTriplane(in_channels=input_channel, out_channels=output_channel, groups=norm_groups)
            )
            upsamplers.append(
                UpsampleTriplane(channels=output_channel, out_channels=output_channel)
            )
            input_channel = output_channel
        
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = nn.ModuleList(upsamplers)
        
        self.conv_out = TriplaneConv(output_channel, out_channels, kernel_size=1, stride=1)
    
    
    def forward(self, feat_planes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_planes: (B, C_in, H_in, W_in)
        Returns:
            feat_planes: (3B, C_out, H_out, W_out)
        """
        feat_planes = self.channel_decoder(feat_planes) # (B, 3C, H, W)
        xy_hidden, xz_hidden, yz_hidden = feat_planes.chunk(3, dim=1)
        feat_planes = torch.cat([xy_hidden, xz_hidden, yz_hidden], dim=0) # (3B, C, H, W)
        
        for resnet, sampler in zip(self.resnets, self.upsamplers):
            feat_planes = resnet(feat_planes, temb=None)
            feat_planes = sampler(feat_planes)
        
        feat_planes = self.conv_out(feat_planes)
        
        return feat_planes


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
        self.blocks = nn.ModuleList(ResnetBlockFC(2*channels, channels) for _ in range(n_blocks))
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
        channels (int): dimension of latent conditioned code c (channels of triplane)
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

        self.point_emb = RFFPointEmbedding(hidden_size, scale=0.9, pooling="mean") # TODO: use PE or not? # self.proj_p = nn.Linear(3, hidden_size)
        self.proj_c = nn.ModuleList([nn.Linear(channels, hidden_size) for i in range(n_blocks)])
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


class TriplaneVAE(nn.Module):
    def __init__(
        self,
        triplane_res: int, # = 128,
        triplane_channels: int, # = 32,
        latent_channels: int, # = 4,
        pt_enc_channels: int,
        pt_enc_num_blocks: int = 5,
        padding: float = 0.1, # conventional padding parameter of OccNet for unit cube
        downsampler_channels: int = 128,
        upsampler_channels: int = 256,
        num_sample_steps: int = 1,
        unet_down_block_types: Tuple[str] = ("DownBlockTriplane", "ResnetDownsampleBlockTriplane"),
        unet_up_block_types: Tuple[str] = ("UpBlockTriplane", "ResnetUpsampleBlockTriplane"),
        unet_block_out_channels: Tuple[int] = (32, 64),
        dec_dim_out: int = 1,
        dec_hidden_size: int = 256,
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
            out_channels=triplane_channels,
            n_blocks=pt_enc_num_blocks,
            triplane_res=triplane_res,
            padding=padding
        )
        
        enc_latent_channels = 2*latent_channels if kl_weight > 0 else latent_channels
        self.enc_unet = UNetTriplaneModel(
            triplane_res=triplane_res,
            in_channels=triplane_channels,
            out_channels=enc_latent_channels,
            down_block_types=unet_down_block_types,
            up_block_types=unet_up_block_types,
            block_out_channels=unet_block_out_channels,
            layers_per_block=1
        )
        
        self.downsampler = DownSampler(
            channels=downsampler_channels,
            in_channels=enc_latent_channels,
            out_channels=enc_latent_channels,
            num_sample_steps=num_sample_steps
        )

        self.upsampler = UpSampler(
            channels=upsampler_channels,
            in_channels=latent_channels,
            out_channels=latent_channels,
            num_sample_steps=num_sample_steps
        )
        
        self.dec_unet = UNetTriplaneModel(
            triplane_res=triplane_res,
            in_channels=latent_channels,
            out_channels=dec_hidden_size,
            down_block_types=unet_down_block_types,
            up_block_types=unet_up_block_types,
            block_out_channels=unet_block_out_channels,
            layers_per_block=1
        )
        
        self.decoder = TriplaneDecoder(
            dim_out=dec_dim_out,
            channels=triplane_channels,
            hidden_size=dec_hidden_size,
            n_blocks=n_blocks,
            padding=padding
        )
        
        
    def encode(self, input_pts: torch.Tensor, deterministic: bool = False):
        feat_planes, feat = self.point_encoder(input_pts) # (3B, C, H, W), (B, N, C)
        triplane = feat_planes.clone() # (3B, C, H, W)
        feat_planes = self.enc_unet(feat_planes).sample
        
        bs, ch, height, width = feat_planes.shape
        assert bs % 3 == 0
        
        feat_planes = self.downsampler(feat_planes)
        
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
        z = self.upsampler(z) # (3B, C, H, W)
        return self.dec_unet(z).sample
    
    
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
    
    # memory usage: 19000 ~ 20000
    config = """
    target: triplane_vae.TriplaneVAE
    params:
        triplane_res: 128
        triplane_channels: 32
        latent_channels: 4
        pt_enc_channels: 32
        pt_enc_num_blocks: 5
        downsampler_channels: 64
        upsampler_channels: 128
        num_sample_steps: 1
        unet_down_block_types: ["DownBlockTriplane", "ResnetDownsampleBlockTriplane"]
        unet_up_block_types: ["UpBlockTriplane", "ResnetUpsampleBlockTriplane"]
        unet_block_out_channels: [32, 64]
        dec_dim_out: 1
        dec_hidden_size: 32
        n_blocks: 5
        # plane_aggregation: "sum"
        kl_weight: 1.0e-6
        scaling_factor: Null
    """
    config = yaml.safe_load(config)
    model = instantiate_from_config(config).cuda()
    
    pts = torch.rand((8, 16384, 3)).cuda() * 2 - 1 # [-1, 1]
    pts_prove = torch.rand((8, 4096, 3)).cuda() * 2 - 1 # [-1, 1]
    res_occ, res_posterior, res_enc_triplane, res_dec_triplane = model(pts, query_pts=pts_prove)
    print("occ.shape: ", res_occ.shape)
    print("latent shape: ", res_posterior.sample().shape)
    print("enc_triplane.shape: ", res_enc_triplane.shape)
    print("dec_triplane.shape: ", res_dec_triplane.shape)
    
    """ Results
    occ.shape:  torch.Size([8, 4096, 1])
    latent shape:  torch.Size([8, 4, 64, 64])
    enc_triplane.shape:  torch.Size([24, 32, 128, 128])
    dec_triplane.shape:  torch.Size([24, 32, 128, 128])
    """


if __name__ == "__main__":
    __test__()
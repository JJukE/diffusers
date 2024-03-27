""" Triplane VAE based on the LatentDiffusion autoencoderKL (https://github.com/CompVis/latent-diffusion) """
from pathlib import Path

import numpy as np
import torch
from torch import nn
from einops import rearrange
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
# from ...src.diffusers import ModelMixin
# from ...src.diffusers.configuration_utils import ConfigMixin, register_to_config


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims
        )

    def mode(self):
        return self.mean


class TriplaneVAE(ModelMixin, ConfigMixin):
    """ (128, 128, 96) → (64, 64, 4) → (128, 128, 96) to use LDM of pre-trained SD """
    
    @register_to_config
    def __init__(
        self,
        aeconfig,
        embed_dim,
        latent_scale_factor = None,
        kl_weight=1.0e-6,
        geom_weight=1.,
        recon_loss_type="l2"
    ):
        super().__init__()
        
        self.latent_scale_factor = latent_scale_factor
        
        self.encoder = Encoder(**aeconfig)
        self.decoder = Decoder(**aeconfig)
        
        self.quant_conv = torch.nn.Conv2d(2*embed_dim, 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)
        self.embed_dim = embed_dim
        
        self.kl_weight = kl_weight
        self.geom_weight = geom_weight
        self.recon_loss_type = recon_loss_type


    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        # print("Shape of latent: ", posterior.sample().shape)
        return posterior


    def decode(self, z):
        latent = z / self.latent_scale_factor if self.latent_scale_factor is not None else z
        latent = self.post_quant_conv(latent)
        dec = self.decoder(latent)
        # print("Shape of decoded triplane: ", dec.shape)
        return dec
    

    def forward(self, triplane_decoder, inputs, coords, gt, stats_dir, sample_posterior=True):
        # assert inputs.shape[0] == len(self.triplane_decoder.embeddings) // 3
        
        # sampling
        rec, post = self.sample(inputs, sample_posterior)
        
        # prepare for decoding the occupancy values
        with torch.no_grad():
            normalized_triplanes = denormalize(rec, stats_dir=stats_dir)
            normalized_triplanes = rearrange(normalized_triplanes, "b (p c) h w -> b p c h w", p=3)
            for batch_idx in range(rec.shape[0]):
                for plane_idx in range(3):
                    triplane_decoder.embeddings[3 * batch_idx + plane_idx][0] = normalized_triplanes[batch_idx][plane_idx]
        
        # reconstruction loss
        if self.recon_loss_type == "l2":
            rec_loss = torch.mean((inputs.contiguous() - rec.contiguous()) ** 2)
        elif self.recon_loss_type == "l1":
            rec_loss = torch.abs(inputs.contiguous() - rec.contiguous())
        else:
            raise NotImplementedError("{} loss are not implemented for the reconstruction loss.".format(self.recon_loss_type))
        
        # kl loss
        kl_loss = post.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss = self.kl_weight * kl_loss # const
        kl_loss = torch.full_like(rec_loss, kl_loss.item()) # same shape to the rec_loss
        
        # geometry loss
        geom_loss = 0.
        preds = torch.zeros_like(gt)
        for i in range(inputs.shape[0]):
            preds[i] = triplane_decoder.pred_occ(i, rearrange(coords[i], "n d -> 1 n d"))[0]
        geom_loss = nn.BCEWithLogitsLoss()(preds, gt) / inputs.shape[0]
        geom_loss = self.geom_weight * geom_loss
        geom_loss = torch.full_like(rec_loss, geom_loss.item()) # same shape to the rec_loss
        
        total_loss = rec_loss + kl_loss + geom_loss
        
        return {
            "loss": total_loss,
            "recon_loss": rec_loss,
            "kl_loss": kl_loss,
            "geom_loss": geom_loss
        }
    
    
    def sample(self, input, sample_posterior=True):
        # print("input shape: ", input.shape)
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample() * self.latent_scale_factor if self.latent_scale_factor is not None else posterior.sample()
        else:
            z = posterior.mode() * self.latent_scale_factor if self.latent_scale_factor is not None else posterior.mode()
        # print("latent shape: ", z.shape)
        dec = self.decode(z)
        # print("recon shape: ", dec.shape)
        return dec, posterior


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(4,2,1,1), num_res_blocks, res_updown_level=(4), dropout=0.,
                 resamp_with_conv=True, in_channels, resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        
        self.ch = ch # try 256? 128?
        self.temb_ch = 0
        self.num_levels = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.res_updown_level = list(res_updown_level)
        self.num_diff_res = len(self.res_updown_level)
        self.in_channels = in_channels
        
        # reducing channels
        self.conv_in = nn.Conv2d(in_channels, ch*ch_mult[0], kernel_size=3, stride=1, padding=1)
        
        curr_res = resolution
        in_ch_mult = (ch_mult[0], ) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult # ex: (4,4,4,2,1)
        self.down = nn.ModuleList()
        for i_level in range(self.num_levels):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if ch_mult[i_level] in self.res_updown_level:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
                self.res_updown_level.remove(ch_mult[i_level])
            self.down.append(down)
        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        num_groups_cand = [32, 16, 8, 4, 2]
        for cand in num_groups_cand:
            if block_in % cand == 0:
                self.norm_out = torch.nn.GroupNorm(num_groups=cand, num_channels=block_in, eps=1e-6, affine=True)
                break
        assert self.norm_out is not None, "GroupNorm layer cannot be defined. Check out the channels."
        # self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
    
    def forward(self, x):
        # timestep embedding
        temb = None
        
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_levels):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                hs.append(h)
            if hasattr(self.down[i_level], "downsample"):
                hs.append(self.down[i_level].downsample(hs[-1]))
        
        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)
        
        # end
        h = self.norm_out(h)
        h = h * torch.sigmoid(h) # nonlinearity
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(4,2,1,1), num_res_blocks, res_updown_level=(4), dropout=0., resamp_with_conv=True,
                 in_channels, resolution, z_channels, give_pre_end=False, tanh_out=False, **ignore_kwargs):
        super().__init__()
        
        self.ch = ch
        self.temb_ch = 0
        self.num_levels = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.res_updown_level = list(res_updown_level)
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (ch_mult[0], ) + tuple(ch_mult) # ex: (4,4,2,1,1)
        block_in = ch * ch_mult[self.num_levels - 1]
        curr_res = resolution // 2 ** len(self.res_updown_level)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        
        # increasing channels
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_levels)):
            block = nn.ModuleList()
            # block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if ch_mult[i_level] in self.res_updown_level:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
                self.res_updown_level.remove(ch_mult[i_level])
            self.up.append(up)
        
        # end
        num_groups_cand = [32, 16, 8, 4, 2]
        for cand in num_groups_cand:
            if block_in % cand == 0:
                self.norm_out = torch.nn.GroupNorm(num_groups=cand, num_channels=block_in, eps=1e-6, affine=True)
                break
        assert self.norm_out is not None, "GroupNorm layer cannot be defined. Check out the channels."
        # self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
    
    def forward(self, z):
        self.last_z_shape = z.shape
        
        # timestep embedding
        temb = None
        
        # z to block_in
        h = self.conv_in(z)
        
        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)
        
        # increasing channels
        for i_level in range(self.num_levels):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)
        
        # end
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = h * torch.sigmoid(h) # nonlinearity
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        
        num_groups_cand = [32, 16, 8, 4, 2]
        self.norm1 = None
        self.norm2 = None
        
        for cand in num_groups_cand:
            if in_channels % cand == 0:
                self.norm1 = torch.nn.GroupNorm(num_groups=cand, num_channels=in_channels, eps=1e-6, affine=True)
                break
        assert self.norm1 is not None, "GroupNorm layer cannot be defined. Check out the channels."
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        
        for cand in num_groups_cand:
            if out_channels % cand == 0:
                self.norm2 = torch.nn.GroupNorm(num_groups=cand, num_channels=out_channels, eps=1e-6, affine=True)
                break
        assert self.norm2 is not None, "GroupNorm layer cannot be assigned."
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = h*torch.sigmoid(h) # nonlinearity
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(temb*torch.sigmoid(temb))[:,:,None,None] # nonlinearity

        h = self.norm2(h)
        h = h*torch.sigmoid(h) # nonlinearity
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


def denormalize(sample, stats_dir):
    stats_dir = Path(stats_dir)
    min_vals = np.load(Path(stats_dir) / "lower_bound.npy").astype(np.float32)
    max_vals = np.load(Path(stats_dir) / "upper_bound.npy").astype(np.float32)
    min_vals = torch.tensor(rearrange(min_vals, "1 c -> 1 c 1 1")).to(sample.device)
    max_vals = torch.tensor(rearrange(max_vals, "1 c -> 1 c 1 1")).to(sample.device)
    sample = ((sample + 1) / 2) * (max_vals - min_vals) + min_vals
    return sample


if __name__ == "__main__":
    aeconfig = {
        "z_channels": 4,
        "resolution": 128,
        "in_channels": 96,
        "out_ch": 96,
        "ch": 256,
        "ch_mult": [4, 4, 2, 1],
        "res_updown_level": [4],
        "num_res_blocks": 2,
        "dropout": 0.
    }
    config = {
        "aeconfig": aeconfig,
        "embed_dim": 4,
        "latent_scale_factor": None,
        "kl_weight": 1.0e-6,
        "geom_weight": 1.,
        "recon_loss_type": "l2",
        "device": "cuda"
    }
    
    model = TriplaneVAE(config=config)
    
    model.save_pretrained("/root/hdd1/t2std/triplane_vae/chair_v24_vae_96_4_ch_256_4421_geom_weight_1/best_ep000395.pth", config=config)
    model.push_to_hub("/root/hdd1/t2std/triplane_vae/chair_v24_vae_96_4_ch_256_4421_geom_weight_1/best_ep000395.pth", config=config)
    model = TriplaneVAE.from_pretrained("JJukE/")
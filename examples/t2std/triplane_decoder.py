import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


class FourierFeatureTransform(nn.Module):
    def __init__(self, num_input_channels, mapping_size, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):
        B, N, C = x.shape
        x = (rearrange(x, "b n c -> (b n) c") @ self._B).contiguous()
        x = (rearrange(x, "(b n) c -> b n c", b=B, n=N)).contiguous()
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class MultiTriplane(nn.Module):
    def __init__(self, num_objs, channels=32, input_dim=3, output_dim=1, noise_val=None, device='cuda'):
        super().__init__()
        self.device = device
        self.num_objs = num_objs
        self.channels = channels
        self.noise_val = noise_val
        
        # Triplane Features
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, channels, 128, 128)*0.001) for _ in range(3*num_objs)]) # (3B, 1, C, H, W)
        
        # Shared MLP - Use this if you want a PE
        self.net = nn.Sequential(
            FourierFeatureTransform(channels, channels*2, scale=1),
            nn.Linear(channels*4, channels*4),
            nn.ReLU(inplace=True),
            
            nn.Linear(channels*4, channels*4),
            nn.ReLU(inplace=True),
            
            nn.Linear(channels*4, output_dim)
        )

    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        # plane → (B, C, H, W)
        coords2d = rearrange(coords2d, "b n d -> b 1 n d").contiguous() # (B, 1, N, 2)
        sampled_features = F.grid_sample(plane, # (B, C, H, W)
                                         coords2d, # (B, 1, N, 2)
                                         mode='bilinear', padding_mode='zeros', align_corners=True) # (B, C, 1, N)
        sampled_features = rearrange(sampled_features, "b c 1 n -> b (1 n) c").contiguous()
        return sampled_features
    
    def pred_occ(self, obj_idx, coordinates):
        """ Predict occupancy value given object index and coordinates """
        batch_size, n_coords, n_dims = coordinates.shape
        
        xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[3*obj_idx+0])
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[3*obj_idx+1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[3*obj_idx+2])
        
        #if self.noise_val != None:
        #    xy_embed = xy_embed + self.noise_val*torch.empty(xy_embed.shape).normal_(mean = 0, std = 0.5).to(self.device)
        #    yz_embed = yz_embed + self.noise_val*torch.empty(yz_embed.shape).normal_(mean = 0, std = 0.5).to(self.device)
        #    xz_embed = xz_embed + self.noise_val*torch.empty(xz_embed.shape).normal_(mean = 0, std = 0.5).to(self.device)
        
        features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        if self.noise_val != None and self.training:
            features = features + self.noise_val*torch.empty(features.shape).normal_(mean = 0, std = 0.5).to(self.device)
        
        return self.net(features)


    def forward(self, obj_idx, coords, gt, tv_weight=1e-2, l2_weight=1e-3, edr_weight=3e-1, multi=False):
        coords, gt = coords.float(), gt.float()
        
        # calculate loss
        preds = self.pred_occ(obj_idx, coords)
        naive_loss = nn.BCEWithLogitsLoss()(preds, gt)
        # naive_loss = nn.functional.mse_loss(preds, gt)
        
        # density regularization
        rand_coords = torch.rand_like(coords) * 2 - 1
        rand_coords_offset = rand_coords + torch.randn_like(rand_coords) * 1e-2
        d_rand_coords = self.pred_occ(obj_idx, rand_coords)
        d_rand_coords_offset = self.pred_occ(obj_idx, rand_coords_offset)
        
        tv_reg = self.tvreg() * tv_weight if multi else None
        l2_reg = self.l2reg() * l2_weight if multi else None
        edr_reg = nn.functional.mse_loss(d_rand_coords, d_rand_coords_offset) * edr_weight
        
        if multi:
            total_loss = naive_loss + tv_reg + l2_reg + edr_reg
            return {
                "loss": total_loss,
                "naive_loss": naive_loss,
                "l2_reg": l2_reg,
                "tv_reg": tv_reg,
                "edr_reg": edr_reg
            }
        else:
            total_loss = naive_loss + edr_reg
            return {
                "loss": total_loss,
                "naive_loss": naive_loss,
                "edr_reg": edr_reg
            }
    
    def tvreg(self):
        l = 0
        for embed in self.embeddings:
            l += ((embed[:, :, 1:] - embed[:, :, :-1])**2).sum()**0.5
            l += ((embed[:, :, :, 1:] - embed[:, :, :, :-1])**2).sum()**0.5
        return l/self.num_objs
    
    def l2reg(self):
        l = 0
        for embed in self.embeddings:
            l += (embed**2).sum()**0.5
        return l/self.num_objs
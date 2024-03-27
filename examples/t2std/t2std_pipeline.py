""" JJukE """
from typing import Dict, Union

import numpy as np
import torch
from einops import rearrange, repeat
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.utils import BaseOutput
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    KarrasVeScheduler,
    UNet2DConditionModel
)
from examples.t2std.triplane_vae import TriplaneVAE


class TriplaneDiffusionOutput(BaseOutput):
    """ Output class for Triplane Diffusion pipeline.

    Args:
        triplane (np.ndarray) (B, C, H, W):
            Predicted triplanes with depth values in the range of [0, 1]. Channel dimension C corresponds
            to (3 * channels_per_plane).
    """

    triplane: np.ndarray


class TriplaneDiffusionPipeline(DiffusionPipeline):
    """ Pipeline for Triplane Diffusion
    
    This model inherits from `DiffusionPipeline`. Check the superclass documentation for the generic methods the
    library implements for all the pipelines such as downloading

    Args:
        DiffusionPipeline (_type_): _description_
    """
    latent_scale_factor = 0.18215
    
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: TriplaneVAE,
        scheduler: DDIMScheduler, # KarrasVeScheduler
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer
    ):
        super().__init__()
        
        self.register_modules(
            unet=unet, vae=vae, scheduler=scheduler, text_encoder=text_encoder, tokenizer=tokenizer
        )
        
        self.empty_text_embed = None
    
    @torch.no_grad()
    def __call__(
        self,
        input_triplane: np.ndarray,
        denoising_steps: int = 10,
        processing_res: int = 768,
        batch_size: int = 0,
        color_map: str = "Spectral", ###
        show_progress_bar: bool = True
    ) -> TriplaneDiffusionOutput:
        """ Funciton invoked when calling the pipeline.

        Args:
            input_triplane (np.ndarray):
                Input Triplane with shape of (B, C, H, W). Here, C is corresponding to (3 * channels_per_plane)
            denoising_steps (int, optional):
                Number of diffusion denoising steps (DDIM) during inference. Defaults to 10.
            processing_res (int, optional):
                Maximum resolution of processing. If set to 0, the model will not resize at all.
            batch_size (int, optional):
                Inference batch size which is not bigger than num_ensemble. If set to 0, the model will
                automatically decide the proper batch size. Defaults to 0.
            show_progress_bar (bool, optional):
                Whether to display a progress bar during the diffusion denoising. Defaults to True.

        Returns:
            TriplaneDiffusionOutput:
                Output class for Traiplne Diffusion pipeline.
        """
        
        device = self.device
        input_size = input_triplane.size
        
        assert processing_res >= 0
        assert denoising_steps >= 1
        
        input_triplane = torch.from_numpy(input_triplane).to(self.dtype).to(device)
        assert input_triplane.min() >= -1. and input_triplane.max() <= 1.
        
        return TriplaneDiffusionOutput(
            triplane=input_triplane
        )
    
    
    @torch.no_grad()
    def single_infer(self, triplane: torch.Tensor, num_inference_steps: int, show_pbar: bool) -> torch.Tensor:
        """ Perform an individual triplane prediction without ensembling. # TODO: check if it is right

        Args:
            triplane (torch.Tensor):
                Input triplane.
            num_inference_steps (int):
                Number of diffusion denoising steps (DDIM) during inference.
            show_pbar (bool):
                Display a progress bar of diffusion denoising.

        Returns:
            torch.Tensor: Predicted triplane features.
        """
        device = triplane.device
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps # (T,)
        
        # TODO: for each plane? for each channel?
        # Encode triplane
        triplane_latent = self.encode_triplane(triplane)
        
        # Initial latent
        latent = torch.randn(triplane_latent.shape, device=device, dtype=self.dtype) # (B, ?, h, w)
        
        # Batched empty text embedding
        if self.empty_text_embed is None:
            prompt = ""
            text_inputs = self.tokenizer(
                prompt,
                padding="do_not_pad",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
            self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)
        batch_empty_text_embed = repeat(self.empty_text_embed, "n c -> b n c", b=triplane_latent.shape[0]) # (B, 2, 1024)
        
        # Denoising loop
        if show_pbar:
            iter = tqdm(enumerate(timesteps), total=len(timesteps), leave=False, desc=" " * 4 + "Diffusion denoising")
        else:
            iter = enumerate(timesteps)
        
        for i, t in iter:
            # unet_input = torch.cat([triplane_xy_latent, triplane_yz_latent, triplane_xz_latent], dim=1) # this order is important
            unet_input = triplane_latent
            
            # Predict the noise residual
            noise_pred = self.unet(unet_input, encoder_hidden_states=batch_empty_text_embed).sample # (B, ?, h, w)
            
            # Compute the previous noisy sample x_t → x_{t-1}
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample
        torch.cuda.empty_cache()
        recon = self.decode_triplane(latent)
        
        # clip prediction
        recon = torch.clip(recon, -1., 1.)
        return recon
    
    
    def encode_triplane(self, input_triplane: torch.Tensor) -> torch.Tensor:
        """ Encode triplane into latent.

        Args:
            input_triplane (torch.Tensor):
                Input triplane to be encoded.

        Returns:
            torch.Tensor: Triplane latent.
        """
        h = self.vae.encoder(input_triplane)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        
        # scale latent
        latent = mean * self.latent_scale_factor
        return latent
    
    
    def decode_latent(self, triplane_latent: torch.Tensor) -> torch.Tensor:
        """ Decode triplane latent into the triplane.

        Args:
            triplane_latent (torch.Tensor):
                Latent to be decoded.

        Returns:
            torch.Tensor: Decoded triplane.
        """
        # scale latent
        triplane_latent = triplane_latent / self.latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(triplane_latent)
        recon = self.vae.decoder(z)
        return recon
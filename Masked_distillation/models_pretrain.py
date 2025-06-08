import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import to_2tuple
from timm.models.vision_transformer import _load_weights
from timm.models.vision_transformer import Block as vitBlock
import math
from utils.pos_embed import *

from mamba_simple import Mamba

from transformers import CLIPVisionModel, ViTModel, CLIPConfig
import torch.nn.functional as F

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import torch.nn as nn
from timm.models.layers import DropPath



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None,
                 flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def resize_pos_embed(x):
    # [256, C] -> [196, C]
    C = x.shape[-1]
    x = x.reshape(1, 16, 16, C).permute(0, 3, 1, 2)
    x = F.interpolate(x, (14, 14), mode='bicubic', align_corners=False)
    x = x.permute(0, 2, 3, 1).reshape(196, C)
    return x


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.mlp = SwiGLU(dim, dim * 4 * 2 // 3)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        hidden_states = hidden_states + self.drop_path(
            self.mixer(self.norm1(hidden_states), inference_params=inference_params))
        hidden_states = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))

        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, expand=1, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out,
                        init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionMamba(nn.Module):
    def __init__(self, 
                img_size=224,
                patch_size=16, 
                stride=16,
                depth=24,
                embed_dim=192,
                dec_embed_dim=512, 
                channels=3, 
                num_classes=1000,
                norm_layer=nn.LayerNorm, 
                decoder_depth=4, 
                mlp_ratio=4.,
                mask_ratio=0.6,
                decoder_num_heads=16,
                norm_pix_loss=False,
                mask_type="attention", target_norm="whiten", loss_type="smoothl1",
                teacher_model="../../model_hub/vision_tower/openai/clip-vit-large-patch14", # replace with your teacher model path
                ssm_cfg=None,
                drop_path_rate=0.2,
                shuffle_rate=0.4,
                rms_norm=False,
                fused_add_norm=False,
                residual_in_fp32=False,
                bimamba_type="none",
                if_devide_out=False,
                init_layer_scale=None,
                initializer_cfg=None,
                **kwargs):
        
        
        factory_kwargs = {"device": kwargs.get('device', None), "dtype": kwargs.get('dtype', None)}
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # Encoder specifics (Mamba blocks)
        self.patch_embed = PatchEmbed(img_size, patch_size, stride, channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.ssr = [x.item() for x in torch.linspace(0, shuffle_rate, depth)]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        
        self.layers = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=1e-5,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                bimamba_type=bimamba_type,
                drop_path=inter_dpr[i],
                if_devide_out=if_devide_out,
                init_layer_scale=init_layer_scale,
                **factory_kwargs,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)


        assert mask_type in ["random", "attention"]
        self.mask_type = mask_type
        assert target_norm in ["none", "l2", "whiten", "bn"]
        self.target_norm = target_norm
        assert loss_type in ["l2", "l1", "smoothl1"]
        self.loss_type = loss_type
        assert "clip" in teacher_model or "dino" in teacher_model
        self.teacher_model_name = teacher_model

        if "clip-vit-base-patch16" in self.teacher_model_name or "dino-vitb16" in self.teacher_model_name:
            target_dim = 768
            teacher_depth = 12
        else:
            target_dim = 1024
            teacher_depth = 24

        # Decoder specifics (MAE-style transformer decoder)
        self.decoder_embed = nn.Linear(embed_dim, dec_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim), requires_grad=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dec_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            vitBlock(dec_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(dec_embed_dim)
        # self.decoder_pred = nn.Linear(dec_embed_dim, patch_size ** 2 * channels, bias=True)
        self.decoder_pred = nn.Linear(dec_embed_dim, target_dim, bias=True)

        # Loss specifics
        self.norm_pix_loss = norm_pix_loss
        self.mask_ratio = mask_ratio

        # Initialize weights
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.initialize_weights()

        if "clip" in self.teacher_model_name:
            Config = CLIPConfig.from_pretrained(self.teacher_model_name)
            self.clip_model = CLIPVisionModel.from_pretrained(self.teacher_model_name, config=Config.vision_config)
            for name, param in self.clip_model.named_parameters():
                param.requires_grad = False
                if "clip-vit-large-patch14" in self.teacher_model_name and "position_embedding" in name:
                    param.data = torch.cat([param.data[:1], resize_pos_embed(param.data[1:])], dim=0)
            if "clip-vit-large-patch14" in self.teacher_model_name:
                self.clip_model.vision_model.embeddings.position_ids = torch.arange(197).expand((1, -1))

        elif "dino" in self.teacher_model_name:
            self.dino_model = ViTModel.from_pretrained(self.teacher_model_name)
            for param in self.dino_model.parameters():
                param.requires_grad = False

    def initialize_weights(self):
    
        # Initialize positional embeddings
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize cls and mask tokens
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # Initialize other weights
        self.apply(self._init_weights)

        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def denormalize(self, images, type="imagenet"):
        # sr_images [B, 3, H, W]
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1).type_as(images)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1).type_as(images)
        return std*images + mean

    def normalize(self, images, type="clip"):
        # images [B, 3, h, w]
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1).type_as(images)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1).type_as(images)
        return (images - mean) / std
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """ Perform random masking on patch embeddings. """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def attention_masking(self, x, mask_ratio, importance):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = importance.to(x.device) # large is keep, small is remove
        
        # sort noise for each sample
        ids_shuffle = torch.multinomial(noise, L, replacement=False)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_dump = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward_encoder(self, x, attentions, inference_params=None):
        """ Forward pass through the encoder with masking. """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        importance = attentions[-1][:, :, 0, 1:].mean(1)
        # x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        x, mask, ids_restore = self.attention_masking(x, self.mask_ratio, importance)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for i, layer in enumerate(self.layers):
            # implement SLWS 
            x = shuffle_forward(x, layer, inference_params=inference_params, 
                                prob = self.ssr[i], training = self.training)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """ Forward pass through the decoder. """
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed

        for layer in self.decoder_blocks:
            x = layer(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]  # Remove cls token

    @torch.no_grad()
    def forward_clip(self, x):
        if "clip-vit-large-patch14" in self.teacher_model_name:
            x = F.interpolate(x, (196, 196), mode='bicubic', align_corners=False)
            
        x = self.normalize(self.denormalize(x))
        input = {
            "pixel_values": x,
            "output_hidden_states": False,
            "output_attentions": True
        }
        outputs = self.clip_model(**input)
        
        last_hidden_state, _, attentions = outputs[0], outputs[1], outputs[2]
        return last_hidden_state[:, 1:, :], last_hidden_state[:, :1, :].detach(), attentions

    @torch.no_grad()
    def forward_dino(self, x):
        input = {
            "pixel_values": x,
            "output_hidden_states": False,
            "output_attentions": True
        }
        outputs = self.dino_model(**input)
        
        last_hidden_state, _, attentions = outputs[0], outputs[1], outputs[2]
        return last_hidden_state[:, 1:, :], last_hidden_state[:, :1, :], attentions


    def forward_loss(self, target, pred, mask):
        B, L, C = target.shape
        mask = 1. - mask

        if self.loss_type == "l2":
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch


        elif self.loss_type == "smoothl1":
            loss = F.smooth_l1_loss(target, pred, reduction='none', beta=1.0)
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            

        elif self.loss_type == "l1":
            loss = (pred - target).abs()
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            
        return loss

    def forward(self, imgs):
        if "clip" in self.teacher_model_name:
            teacher_patch, teacher_cls, attentions = self.forward_clip(imgs)
        elif "dino" in self.teacher_model_name:
            teacher_patch, teacher_cls, attentions = self.forward_dino(imgs)

        if self.target_norm == "l2":
            teacher_patch = F.normalize(teacher_patch, dim=-1)
        elif self.target_norm == "whiten":
            teacher_patch = F.layer_norm(teacher_patch, (teacher_patch.shape[-1],))
        elif self.target_norm == "bn":
            teacher_patch = (teacher_patch - teacher_patch.mean()) / (teacher_patch.var() + 1.e-6)**.5

        latent, mask, ids_restore = self.forward_encoder(imgs, attentions)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(teacher_patch, pred, mask)
        return loss, pred, mask


@register_model
def arm_base_pz16(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, img_size=224, embed_dim=768, depth=12, dec_embed_dim=512, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        if_abs_pos_embed=True, bimamba_type="v2", **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def arm_large_pz16(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, img_size=224, embed_dim=1024, depth=24, dec_embed_dim=512, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        if_abs_pos_embed=True, bimamba_type="v2", **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def arm_huge_pz16(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, img_size=224, embed_dim=1536, depth=24, dec_embed_dim=512, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        if_abs_pos_embed=True, bimamba_type="v2", **kwargs)
    model.default_cfg = _cfg()
    return model


def shuffle_forward(x, layer: nn.Module, inference_params=None, prob: float = 0.0, training: bool = False):
    """
    Forward pass with optional shuffling of the sequence dimension.

    Args:
    - x (torch.Tensor): Input tensor with shape (B, L, d).
    - residual: Input tensor of the same size of x, required by mamba model
    - layer (nn.Module): A PyTorch module through which x should be passed.
    - prob (float): Probability of shuffling the sequence dimension L.
    - training (bool): Indicates whether the model is in training mode.

    Returns:
    - torch.Tensor: Output tensor from layer, with the sequence dimension
                    potentially shuffled and then restored.
    """
    
    B, L, _ = x.shape
    if training and torch.rand(1).item() < prob:
        # Generate a random permutation of indices
        shuffled_indices = torch.randperm(L, device=x.device).repeat(B, 1)
        # Get inverse indices by sorting the shuffled indices
        inverse_indices = torch.argsort(shuffled_indices, dim=1)

        # Apply the permutation to shuffle the sequence
        x_permuted = x.gather(1, shuffled_indices.unsqueeze(-1).expand(-1, -1, x.size(2)))
        
        # Forward pass through the layer
        output_permuted = layer(x_permuted, inference_params=inference_params)
        # Restore the original order
        output = output_permuted.gather(1, inverse_indices.unsqueeze(-1).expand(-1, -1, output_permuted.size(2)))
    else:
        # Forward pass without shuffling
        output = layer(x, inference_params=inference_params)

    return output
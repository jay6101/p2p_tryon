'''DiT Linformer model for Pytorch with Multi-Scale Cross-Attention for Virtual Try-On

Author: Emilio Morales (mil.mor.mor@gmail.com)
      Dec 2023 (modified)
Further Modified: Jay Sawant – added multi-scale cross-attention
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import Attention


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context):
        b, n, d = x.shape
        h = self.num_heads
        q = self.to_q(x).reshape(b, n, h, d // h).permute(0, 2, 1, 3)
        k = self.to_k(context).reshape(b, -1, h, d // h).permute(0, 2, 1, 3)
        v = self.to_v(context).reshape(b, -1, h, d // h).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, d)
        return self.to_out(out)


class MultiScaleCrossAttention(nn.Module):
    """
    Cross-attends to garment tokens at multiple spatial scales: full, half, and quarter.
    """
    def __init__(self, dim, heads):
        super().__init__()
        # one CrossAttention module per scale
        self.scale_attns = nn.ModuleList([
            CrossAttention(dim, heads),    # full-res
            CrossAttention(dim, heads),    # half-res
            CrossAttention(dim, heads),    # quarter-res
        ])

    def forward(self, x, contexts):
        # contexts: [G_full, G_half, G_quarter]
        for attn, C in zip(self.scale_attns, contexts):
            x = x + attn(x, C)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, rate=0.0):
        super().__init__()
        # Self-attention
        self.ln_1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads=heads, qkv_bias=True)
        # Cross-attention (multi-scale)
        self.ln_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ms_cross_attn = MultiScaleCrossAttention(dim, heads)
        # MLP
        self.ln_2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(rate),
        )
        # Modulation layers
        self.gamma_1 = nn.Linear(dim, dim)
        self.beta_1 = nn.Linear(dim, dim)
        self.scale_1 = nn.Linear(dim, dim)
        self.gamma_cross = nn.Linear(dim, dim)
        self.beta_cross = nn.Linear(dim, dim)
        self.scale_cross = nn.Linear(dim, dim)
        self.gamma_2 = nn.Linear(dim, dim)
        self.beta_2 = nn.Linear(dim, dim)
        self.scale_2 = nn.Linear(dim, dim)
        self._init_weights([
            self.gamma_1, self.beta_1, self.scale_1,
            self.gamma_cross, self.beta_cross, self.scale_cross,
            self.gamma_2, self.beta_2, self.scale_2
        ])

    def _init_weights(self, layers):
        for l in layers:
            nn.init.zeros_(l.weight)
            nn.init.zeros_(l.bias)

    def forward(self, x, t_emb, cloth_tokens, garment_contexts):
        # 1. Self-attention with cloth-agnostic tokens
        combined = torch.cat([cloth_tokens, x], dim=1)
        scale1 = self.gamma_1(t_emb)
        shift1 = self.beta_1(t_emb)
        gate1 = self.scale_1(t_emb).unsqueeze(1)
        msa_out = self.attn(modulate(self.ln_1(combined), shift1, scale1))
        msa_out = msa_out[:, cloth_tokens.size(1):]  # keep only noisy-token outputs
        x = msa_out * gate1 + x

        # 2. Multi-scale cross-attention
        scale_c = self.gamma_cross(t_emb)
        shift_c = self.beta_cross(t_emb)
        gate_c = self.scale_cross(t_emb).unsqueeze(1)
        x_norm = modulate(self.ln_cross(x), shift_c, scale_c)
        cross_out = self.ms_cross_attn(x_norm, garment_contexts)
        x = cross_out * gate_c + x

        # 3. MLP
        scale2 = self.gamma_2(t_emb)
        shift2 = self.beta_2(t_emb)
        gate2 = self.scale_2(t_emb).unsqueeze(1)
        mlp_out = self.mlp(modulate(self.ln_2(x), shift2, scale2))
        return mlp_out * gate2 + x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.ln_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        self._init_weights([self.linear, self.gamma, self.beta])

    def _init_weights(self, layers):
        for l in layers:
            nn.init.zeros_(l.weight)
            nn.init.zeros_(l.bias)

    def forward(self, x, t_emb):
        scale = self.gamma(t_emb)
        shift = self.beta(t_emb)
        x = modulate(self.ln_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    def __init__(self, img_size, dim=64, patch_size=4,
                 depth=3, heads=4, mlp_dim=512, in_channels=3):
        super().__init__()
        # compute patch count
        self.dim = dim
        self.patch_size = patch_size
        H, W = img_size
        self.n_patches = (H // patch_size) * (W // patch_size)
        # positional
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, dim))
        # patch embedding convs
        self.noisy_patches = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.cloth_agnostic_patches = nn.Conv2d(in_channels+1, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.garment_patches = nn.Conv2d(in_channels+1, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        # transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, rate=0.0)
            for _ in range(depth)
        ])
        # time embedding
        self.emb = nn.Sequential(
            PositionalEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # final and pixel shuffle
        self.final = FinalLayer(dim, patch_size, in_channels)
        self.ps = nn.PixelShuffle(patch_size)

    def forward(self, x, t, cloth_agnostic, garment):
        # time embedding
        t_emb = self.emb(t)
        # tokenize
        x_p = self.noisy_patches(x)
        c_p = self.cloth_agnostic_patches(cloth_agnostic)
        g_p = self.garment_patches(garment)
        B, C, H, W = x_p.shape
        # reshape to (B, N, dim)
        x_tokens = x_p.permute(0,2,3,1).reshape(B, -1, C)
        c_tokens = c_p.permute(0,2,3,1).reshape(B, -1, C)
        g_tokens = g_p.permute(0,2,3,1).reshape(B, -1, C)
        # add positional
        x_tokens = x_tokens + self.pos_embedding
        c_tokens = c_tokens + self.pos_embedding
        g_tokens = g_tokens + self.pos_embedding
        # compute multi-scale garment contexts
        G_full = g_tokens  # (B, N, dim)
        G_half = F.avg_pool1d(G_full.transpose(1,2), kernel_size=2, stride=2).transpose(1,2)
        G_quarter = F.avg_pool1d(G_full.transpose(1,2), kernel_size=4, stride=4).transpose(1,2)
        garment_contexts = [G_quarter, G_half, G_full]
        # transformer
        for blk in self.transformer:
            x_tokens = blk(x_tokens, t_emb, c_tokens, garment_contexts)
        # final projector
        x_out = self.final(x_tokens, t_emb).permute(0,2,1)
        x_out = x_out.reshape(B, -1, H, W)
        x_out = self.ps(x_out)
        return x_out

'''DiT Linformer model for Pytorch.

Author: Emilio Morales (mil.mor.mor@gmail.com)
        Dec 2023
Modified for Virtual Try-on with multiple conditions
'''
import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention


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


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

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

class PoseEncoder(nn.Module):
    def __init__(self, pose_dim, dim):
        super().__init__()
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, pose):
        return self.pose_encoder(pose)

class TransformerBlock(nn.Module):
    def __init__(self, seq_len, dim, heads, mlp_dim, k, pose_dim=150, rate=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads=heads, qkv_bias=True)
        self.ln_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(dim, heads)
        self.ln_2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(rate),
        )
        
        # Pose encoder for each block
        self.garment_pose_encoder = PoseEncoder(pose_dim, dim)
        self.target_pose_encoder = PoseEncoder(pose_dim, dim)
        
        # Modulation parameters
        self.gamma_1 = nn.Linear(dim, dim)
        self.beta_1 = nn.Linear(dim, dim)
        self.gamma_cross = nn.Linear(dim, dim)
        self.beta_cross = nn.Linear(dim, dim)
        self.gamma_2 = nn.Linear(dim, dim)
        self.beta_2 = nn.Linear(dim, dim)
        self.scale_1 = nn.Linear(dim, dim)
        self.scale_cross = nn.Linear(dim, dim)
        self.scale_2 = nn.Linear(dim, dim)

        # Initialize weights to zero for modulation and gating layers
        self._init_weights([self.gamma_1, self.beta_1, self.gamma_cross, self.beta_cross,
                          self.gamma_2, self.beta_2, self.scale_1, self.scale_cross, self.scale_2])

    def _init_weights(self, layers):
        for layer in layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, c, cloth_agnostic_tokens, garment_tokens, garment_pose, target_pose):
        # Process poses
        garment_pose_emb = self.garment_pose_encoder(garment_pose)
        target_pose_emb = self.target_pose_encoder(target_pose)
        
        # Concatenate poses with garment tokens
        # Reshape to match sequence length of garment tokens
        b, n, d = garment_tokens.shape
        pose_emb = torch.cat([garment_pose_emb.unsqueeze(1), target_pose_emb.unsqueeze(1)], dim=1)
        garment_context = torch.cat([garment_tokens, pose_emb.expand(b, 2, d)], dim=1)
        
        # Self-attention with cloth agnostic tokens
        # Detach cloth_agnostic_tokens from the computation graph
        combined_tokens = torch.cat([cloth_agnostic_tokens, x], dim=1)
        
        # Self-attention modulation
        scale_msa = self.gamma_1(c)
        shift_msa = self.beta_1(c)
        gate_msa = self.scale_1(c).unsqueeze(1)
        
        # Apply self-attention
        attn_output = self.attn(modulate(self.ln_1(combined_tokens), shift_msa, scale_msa))
        
        # Extract only the output corresponding to the noisy tokens
        attn_output = attn_output[:, cloth_agnostic_tokens.size(1):]
        x = attn_output * gate_msa + x
        
        # Cross-attention modulation
        scale_cross = self.gamma_cross(c)
        shift_cross = self.beta_cross(c)
        gate_cross = self.scale_cross(c).unsqueeze(1)
        
        # Apply cross-attention
        cross_output = self.cross_attn(
            modulate(self.ln_cross(x), shift_cross, scale_cross),
            garment_context
        )
        x = cross_output * gate_cross + x
        
        # MLP modulation
        scale_mlp = self.gamma_2(c)
        shift_mlp = self.beta_2(c)
        gate_mlp = self.scale_2(c).unsqueeze(1)
        
        # Apply MLP
        mlp_output = self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp))
        return mlp_output * gate_mlp + x

class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.ln_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)

        # Initialize weights and biases to zero for modulation and linear layers
        self._init_weights([self.linear, self.gamma, self.beta])

    def _init_weights(self, layers):
        for layer in layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)           

    def forward(self, x, c):
        scale = self.gamma(c)
        shift = self.beta(c)
        x = modulate(self.ln_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    def __init__(self, img_size, dim=64, patch_size=4,
                 depth=3, heads=4, mlp_dim=512, k=64, in_channels=3,
                 pose_dim=150):
        super(DiT, self).__init__()
        self.dim = dim
        self.n_patches = (img_size[0] // patch_size)*(img_size[1] // patch_size)
        self.depth = depth
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches, dim))
        
        # Tokenizer for all images (noisy, cloth agnostic, garment)
        self.patches = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, 
                      stride=patch_size, padding=0, bias=False),
        )
        
        self.transformer = nn.ModuleList()
        for i in range(self.depth):
            self.transformer.append(
                TransformerBlock(
                    self.n_patches, dim, heads, mlp_dim, k, pose_dim=pose_dim
                )
            )

        self.emb = nn.Sequential(
            PositionalEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.final = FinalLayer(dim, patch_size, in_channels)
        self.ps = nn.PixelShuffle(patch_size)

    def forward(self, x, t, cloth_agnostic, garment, garment_pose, target_pose):
        t = self.emb(t)
        
        # Tokenize all images
        x_tokens = self.patches(x)
        cloth_agnostic_tokens = self.patches(cloth_agnostic)
        garment_tokens = self.patches(garment)
        
        # Reshape tokens
        B, C, H, W = x_tokens.shape
        x_tokens = x_tokens.permute([0, 2, 3, 1]).reshape([B, H * W, C])
        cloth_agnostic_tokens = cloth_agnostic_tokens.permute([0, 2, 3, 1]).reshape([B, H * W, C])
        garment_tokens = garment_tokens.permute([0, 2, 3, 1]).reshape([B, H * W, C])
        
        # Add positional embedding to noisy tokens
        x_tokens = x_tokens + self.pos_embedding
        
        # Apply transformer blocks
        for layer in self.transformer:
            x_tokens = layer(x_tokens, t, cloth_agnostic_tokens, garment_tokens, garment_pose, target_pose)

        # Final layer
        x = self.final(x_tokens, t).permute([0, 2, 1])
        x = x.reshape([B, -1, H, W])
        x = self.ps(x)
        return x



# class LinformerAttention(nn.Module):
#     def __init__(self, seq_len, dim, n_heads, k, bias=True):
#         super().__init__()
#         self.n_heads = n_heads
#         self.scale = (dim // n_heads) ** -0.5
#         self.qw = nn.Linear(dim, dim, bias = bias)
#         self.kw = nn.Linear(dim, dim, bias = bias)
#         self.vw = nn.Linear(dim, dim, bias = bias)

#         self.E = nn.Parameter(torch.randn(seq_len, k))
#         self.F = nn.Parameter(torch.randn(seq_len, k))

#         self.ow = nn.Linear(dim, dim, bias = bias)

#     def forward(self, x):
#         q = self.qw(x)
#         k = self.kw(x)
#         v = self.vw(x)

#         B, L, D = q.shape
#         q = torch.reshape(q, [B, L, self.n_heads, -1])
#         q = torch.permute(q, [0, 2, 1, 3])
#         k = torch.reshape(k, [B, L, self.n_heads, -1])
#         k = torch.permute(k, [0, 2, 3, 1])
#         v = torch.reshape(v, [B, L, self.n_heads, -1])
#         v = torch.permute(v, [0, 2, 3, 1])
#         k = torch.matmul(k, self.E[:L, :])

#         v = torch.matmul(v, self.F[:L, :])
#         v = torch.permute(v, [0, 1, 3, 2])

#         qk = torch.matmul(q, k) * self.scale
#         attn = torch.softmax(qk, dim=-1)

#         v_attn = torch.matmul(attn, v)
#         v_attn = torch.permute(v_attn, [0, 2, 1, 3])
#         v_attn = torch.reshape(v_attn, [B, L, D])

#         x = self.ow(v_attn)
#         return x
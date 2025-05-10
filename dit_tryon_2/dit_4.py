'''DiT Transformer model for PyTorch with hierarchical transformer stages, multi-scale skip‐fusion, and high‐frequency residual preservation.  
Maintains same input/output signature: forward(x, t, cloth_agnostic, garment) -> generated image. Requires PyTorch >=1.10.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class SinCosPositionEmbedding2D(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"
        pe = torch.zeros(dim, h, w)
        half = dim // 2
        y = torch.arange(h, dtype=torch.float32).unsqueeze(1)
        x = torch.arange(w, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, half, 2, dtype=torch.float32) *
                        -(math.log(10000.0) / half))
        pe[0:half:2, :, :] = torch.sin(y * div).unsqueeze(2).repeat(1,1,w)
        pe[1:half:2, :, :] = torch.cos(y * div).unsqueeze(2).repeat(1,1,w)
        pe[half::2, :, :]   = torch.sin(x * div).unsqueeze(2).repeat(1,1,h).permute(0,2,1)
        pe[half+1::2, :, :] = torch.cos(x * div).unsqueeze(2).repeat(1,1,h).permute(0,2,1)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):  # x: [B, N, C]
        B, N, C = x.shape
        pe = self.pe.view(1, C, -1).permute(0, 2, 1)
        return pe.expand(B, N, C)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context):
        b,n,d = x.shape; h = self.num_heads
        q = self.to_q(x).view(b,n,h,d//h).permute(0,2,1,3)
        k = self.to_k(context).view(b,-1,h,d//h).permute(0,2,1,3)
        v = self.to_v(context).view(b,-1,h,d//h).permute(0,2,1,3)
        attn = (q @ k.transpose(-1,-2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.permute(0,2,1,3).reshape(b,n,d)
        return self.to_out(out)

class TransformerStage(nn.Module):
    def __init__(self, dim, heads, mlp_dim, depth, rate):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(dim),
                'self_attn': Attention(dim, num_heads=heads),
                'ln2': nn.LayerNorm(dim),
                'cross_attn_garment': CrossAttention(dim, heads),
                'ln3': nn.LayerNorm(dim),
                'cross_attn_cloth': CrossAttention(dim, heads),
                'ln4': nn.LayerNorm(dim),
                'mlp': nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(rate), nn.Linear(mlp_dim, dim), nn.Dropout(rate)),
                'time_mlp': nn.Sequential(nn.Linear(dim, dim*2), nn.SiLU(), nn.Dropout(rate)),
            }) for _ in range(depth)
        ])

    def forward(self, x, t_emb, cloth, garment):
        for blk in self.blocks:
            # self-attention
            h = blk['ln1'](x)
            x = x + blk['self_attn'](h)
            
            # cross-attention with garment
            h2 = blk['ln2'](x)
            x = x + blk['cross_attn_garment'](h2, garment)
            
            # cross-attention with cloth-agnostic
            h3 = blk['ln3'](x)
            x = x + blk['cross_attn_cloth'](h3, cloth)
            
            # time embedding modulation
            time_emb = blk['time_mlp'](t_emb)
            time_scale, time_shift = time_emb.chunk(2, dim=1)
            
            # MLP with time modulation
            h4 = blk['ln4'](x)
            x = x + modulate(blk['mlp'](h4), time_shift, time_scale)
        return x

class DiT(nn.Module):
    def __init__(self, img_size, dim=96, patch_size=4, depth=[2,2,4], heads=[4,8,16], mlp_dim=384, in_ch=3):
        super().__init__()
        self.patch_size = patch_size
        H, W = img_size[0]//patch_size, img_size[1]//patch_size
        self.n_patches = H*W
        # multi-scale linear projections
        self.proj = nn.Conv2d(in_ch, dim, patch_size, patch_size)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        # positional embed
        # self.pos_embed = SinCosPositionEmbedding2D(dim, H, W)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, dim))

        self.emb = nn.Sequential(
            PositionalEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # hierarchical stages
        self.stages = nn.ModuleList([
            TransformerStage(dim*(2**i), heads[i], mlp_dim*(2**i), depth[i], rate=0.1)
            for i in range(len(depth))
        ])
        # patch merging
        self.merge = nn.ModuleList([
            nn.Conv2d(dim*(2**i), dim*(2**(i+1)), kernel_size=2, stride=2)
            for i in range(len(depth)-1)
        ])
        # final output head
        self.to_img = nn.Sequential(
            nn.Conv2d(dim*(2**(len(depth)-1)), dim, 3, padding=1),
            nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, in_ch, 3, padding=1)
        )
        # high-frequency extractor
        self.hf_filter = nn.Conv2d(in_ch+1, in_ch, kernel_size=3, padding=1, bias=False)
        # initialize as Laplacian
        lap = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32)
        self.hf_filter.weight.data = lap.unsqueeze(0).unsqueeze(0).repeat(in_ch,1,1,1)

    def forward(self, x, t, cloth, garment):
        # extract HF from skips
        hf_cloth = self.hf_filter(cloth)
        hf_garment = self.hf_filter(garment)
        # initial projection
        out = self.proj(x)
        c = self.proj(cloth)
        g = self.proj(garment)
        B,C,H,W = out.shape
        # to tokens
        out = out.flatten(2).transpose(1,2)
        c   = c.flatten(2).transpose(1,2)
        g   = g.flatten(2).transpose(1,2)
        pos = self.pos_embed
        out = out + pos; c = c + pos; g = g + pos
        # t_emb = self.time_mlp(t)
        t_emb = self.emb(t)
        # hierarchical processing
        for i, stage in enumerate(self.stages):
            out = stage(out, t_emb, c, g)
            if i < len(self.merge):
                # unflatten to feature map
                fmap = out.transpose(1,2).view(B, C*(2**i), H//(2**i), W//(2**i))
                fmap = self.merge[i](fmap)
                # repack tokens
                B2,C2,H2,W2 = fmap.shape
                out = fmap.flatten(2).transpose(1,2)
        # final unflatten
        fmap = out.transpose(1,2).view(B, C*(2**(len(self.stages)-1)), H//(2**(len(self.stages)-1)), W//(2**(len(self.stages)-1)))
        # reconstruct image + refine
        img = self.to_img(fmap)
        # fuse HF details
        img = img + F.interpolate(hf_cloth, size=img.shape[-2:]) + F.interpolate(hf_garment, size=img.shape[-2:])
        return img

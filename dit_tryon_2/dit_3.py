'''DiT Transformer model for PyTorch with skip‐fusion to preserve high‐frequency garment and cloth details.  
Maintains same input/output signature: forward(x, t, cloth_agnostic, garment) -> generated image.'''  
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

class SinCosPositionEmbedding2D(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        assert dim % 4 == 0, "Embedding dimension must be divisible by 4"
        self.dim = dim
        self.h = h
        self.w = w
        # create constant positional encoding
        pe = torch.zeros(dim, h, w)
        half = dim // 2
        div_term = torch.exp(torch.arange(0, half, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / half))
        
        # Create position indices
        y_pos = torch.arange(h, dtype=torch.float32).unsqueeze(1).unsqueeze(0)  # [1, h, 1]
        x_pos = torch.arange(w, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, w]
        
        # Reshape div_term for broadcasting
        div_term = div_term.unsqueeze(1).unsqueeze(1)  # [half/2, 1, 1]
        
        # Calculate embeddings - properly shaped for broadcasting
        # First quarter: sin(y)
        pos_y_sin = torch.sin(y_pos * div_term)  # [half/2, h, 1]
        pos_y_cos = torch.cos(y_pos * div_term)  # [half/2, h, 1]
        pos_x_sin = torch.sin(x_pos * div_term)  # [half/2, 1, w]
        pos_x_cos = torch.cos(x_pos * div_term)  # [half/2, 1, w]
        
        # Handle y embeddings (first half of dim)
        for i in range(0, half//2):
            pe[i*2, :, :] = pos_y_sin[i].expand(h, w)
            pe[i*2+1, :, :] = pos_y_cos[i].expand(h, w)
            
        # Handle x embeddings (second half of dim)
        for i in range(0, half//2):
            pe[half + i*2, :, :] = pos_x_sin[i].expand(h, w)
            pe[half + i*2+1, :, :] = pos_x_cos[i].expand(h, w)
            
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, C, H, W]

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        pe = self.pe.view(1, C, self.h * self.w).permute(0, 2, 1)  # [1, N, C]
        return pe.expand(B, -1, -1)



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
        b, n, d = x.shape
        h = self.num_heads
        q = self.to_q(x).view(b, n, h, d//h).permute(0,2,1,3)
        k = self.to_k(context).view(b, -1, h, d//h).permute(0,2,1,3)
        v = self.to_v(context).view(b, -1, h, d//h).permute(0,2,1,3)
        attn = (q @ k.transpose(-1,-2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.permute(0,2,1,3).reshape(b, n, d)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, rate=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=heads, qkv_bias=True)
        self.ln_cross = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = CrossAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(rate)
        )
        # FiLM params
        self.gamma_cross = nn.Linear(dim, dim)
        self.beta_cross  = nn.Linear(dim, dim)
        self.gate_cross  = nn.Linear(dim, dim)
        self.gamma_mlp   = nn.Linear(dim, dim)
        self.beta_mlp    = nn.Linear(dim, dim)
        self.gate_mlp    = nn.Linear(dim, dim)
        nn.init.zeros_(self.gate_cross.weight); nn.init.ones_(self.gate_cross.bias)
        nn.init.zeros_(self.gate_mlp.weight);   nn.init.ones_(self.gate_mlp.bias)

    def forward(self, x, t_emb, cloth_tokens, garment_tokens):
        combined_tokens = torch.cat([cloth_tokens, x], dim=1)
        h = self.ln1(combined_tokens)
        attn_output = self.attn(h)
        x = x + attn_output[:, cloth_tokens.size(1):]
        # cross-attn
        shift_c = self.beta_cross(t_emb)
        scale_c = self.gamma_cross(t_emb)
        gate_c  = torch.sigmoid(self.gate_cross(t_emb)).unsqueeze(1)
        cross_in  = modulate(self.ln_cross(x), shift_c, scale_c)
        cross_out = self.cross_attn(cross_in, garment_tokens)
        x = x + cross_out * gate_c
        # mlp
        shift_m = self.beta_mlp(t_emb); scale_m = self.gamma_mlp(t_emb)
        gate_m  = torch.sigmoid(self.gate_mlp(t_emb)).unsqueeze(1)
        mlp_in  = modulate(self.ln2(x), shift_m, scale_m)
        x = x + self.mlp(mlp_in) * gate_m
        return x

class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_ch):
        super().__init__()
        self.ln   = nn.LayerNorm(dim, eps=1e-6)
        self.proj = nn.Linear(dim, patch_size*patch_size*out_ch)
        self.gamma = nn.Linear(dim, dim); self.beta = nn.Linear(dim, dim)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, x, t_emb):
        scale = self.gamma(t_emb); shift = self.beta(t_emb)
        x = modulate(self.ln(x), shift, scale)
        return self.proj(x)

class DiT(nn.Module):
    def __init__(self, img_size, dim=64, patch_size=4,
                 depth=4, heads=8, mlp_dim=256, in_channels=3, rate=0.1):
        super().__init__()
        self.dim = dim; self.patch_size = patch_size
        H, W = img_size[0]//patch_size, img_size[1]//patch_size
        self.n_patches = H * W
        self.pos_embed = SinCosPositionEmbedding2D(dim, H, W)
        #self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, dim))
        self.emb = nn.Sequential(
            PositionalEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # tokenizers
        self.noisy_patches  = nn.Conv2d(in_channels,    dim, patch_size, patch_size)
        self.cloth_patches  = nn.Conv2d(in_channels+1,  dim, patch_size, patch_size)
        self.garment_patches= nn.Conv2d(in_channels+1,  dim, patch_size, patch_size)
        # time embed
        # self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        # transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, rate)
            for _ in range(depth)
        ])
        # final
        self.final = FinalLayer(dim, patch_size, in_channels)
        # pixelshuffle + decoder
        # updated conv_decoder to accept skip-fusion of generated, cloth and garment inputs
        skip_ch = in_channels + (in_channels+1)*2
        self.ps = nn.PixelShuffle(patch_size)
        self.conv_decoder = nn.Sequential(
            nn.Conv2d(skip_ch, dim, 3, padding=1),
            nn.GroupNorm(8, dim), nn.SiLU(),
            nn.Conv2d(dim, in_channels, 3, padding=1)
        )

    def forward(self, x, t, cloth_agnostic, garment):
        t_emb = self.emb(t)
        # encode tokens
        x_tok = self.noisy_patches(x)
        c_tok = self.cloth_patches(cloth_agnostic)
        g_tok = self.garment_patches(garment)
        B,C,H,W = x_tok.shape
        # flatten
        x_tok = x_tok.permute(0,2,3,1).reshape(B, self.n_patches, C)
        c_tok = c_tok.permute(0,2,3,1).reshape(B, self.n_patches, C)
        g_tok = g_tok.permute(0,2,3,1).reshape(B, self.n_patches, C)
        # add positional
        pos = self.pos_embed(x_tok)
        x_tok = x_tok + pos; c_tok = c_tok + pos; g_tok = g_tok + pos
        # transformer
        for blk in self.blocks:
            x_tok = blk(x_tok, t_emb, c_tok, g_tok)
        # project to patches
        out = self.final(x_tok, t_emb)
        out = out.permute(0,2,1).reshape(B, -1, H, W)
        out = self.ps(out)
        # skip-fusion: concat generated, cloth_agnostic and garment to preserve high-frequency details
        skip = torch.cat([out, cloth_agnostic, garment], dim=1)
        out = self.conv_decoder(skip)
        return out

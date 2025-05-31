import torch
import torch.nn as nn
from omegaconf import DictConfig
import torch.nn.functional as F

def get_model(cfg: DictConfig):
    # Create model based on configuration
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
    if cfg.model.type == "simple_cnn":
        model = SimpleCNN(**model_kwargs)
    elif cfg.model.type == "unet":
        model = UNet(**model_kwargs)
    elif cfg.model.type == "vit_unet":
        model = ViT_UNet(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model


# --- Model Architectures ---


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.skip(identity)
        out = self.relu(out)

        return out


class SimpleCNN(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        kernel_size=3,
        init_dim=64,
        depth=4,
        dropout_rate=0.2,
    ):
        super().__init__()

        # Initial convolution to expand channels
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
        )

        # Residual blocks with increasing feature dimensions
        self.res_blocks = nn.ModuleList()
        current_dim = init_dim

        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim
            self.res_blocks.append(ResidualBlock(current_dim, out_dim))
            if i < depth - 1:  # Don't double the final layer
                current_dim *= 2

        # Final prediction layers
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final = nn.Sequential(
            nn.Conv2d(current_dim, current_dim // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(current_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_dim // 2, n_output_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.initial(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.dropout(x)
        x = self.final(x)

        return x

"""
UNET
"""
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, dropout_rate=0.4, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_input_channels
        self.n_classes = n_output_channels
        self.bilinear = bilinear

        self.dropout = nn.Dropout2d(dropout_rate)
        self.inc = (DoubleConv(n_input_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_output_channels))

    def forward(self, x):
        x1 = self.inc(x)
        
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.dropout(x)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = nn.utils.checkpoint(self.inc)
        self.down1 = nn.utils.checkpoint(self.down1)
        self.down2 = nn.utils.checkpoint(self.down2)
        self.down3 = nn.utils.checkpoint(self.down3)
        self.down4 = nn.utils.checkpoint(self.down4)
        self.up1 = nn.utils.checkpoint(self.up1)
        self.up2 = nn.utils.checkpoint(self.up2)
        self.up3 = nn.utils.checkpoint(self.up3)
        self.up4 = nn.utils.checkpoint(self.up4)
        self.outc = nn.utils.checkpoint(self.outc)


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

# Helper function for 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

# Initial Convolutional Block (Inc Layer)
class Inc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Convolutional Block Attention Module (CBAM)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel_size, padding=spatial_kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        avg_pool = F.avg_pool2d(x, x.size(2)).view(x.size(0), x.size(1), 1, 1)
        max_pool = F.max_pool2d(x, x.size(2)).view(x.size(0), x.size(1), 1, 1)
        channel_att_avg = self.channel_attention(avg_pool)
        channel_att_max = self.channel_attention(max_pool)
        channel_att = channel_att_avg + channel_att_max
        x = x * channel_att.expand_as(x)

        # Spatial Attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        x = x * spatial_att
        return x

# Multi-Head Self-Attention (MSA) from ViT
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# FeedForward Neural Network (FFN)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# Vision Transformer Block (Encoder part)
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

# Multiple Hierarchies Attention Module (MHAM)
class MHAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        att_weights = self.se_block(x)
        x_att = x * att_weights
        return x_att

# ViT 2D Block for handling feature maps
class ViT_2D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, num_transformer_blocks, transformer_dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.patch_size = patch_size
        self.transformer_dim = transformer_dim
        self.out_channels = out_channels
        
        # Linear projection from patch features to transformer_dim
        self.patch_embedding = nn.Linear((patch_size**2) * in_channels, transformer_dim)
        
        self.register_buffer('pos_embedding_base', torch.randn(1, (128 // patch_size)**2, transformer_dim))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(transformer_dim, heads, dim_head, mlp_dim)
            for _ in range(num_transformer_blocks)
        ])

        # Linear projection from transformer_dim back to patch features
        self.patch_to_feature = nn.Linear(transformer_dim, (patch_size**2) * out_channels)
        self.norm_output = nn.LayerNorm((patch_size**2) * out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        # Patchify and embed
        x_patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x_tokens = self.patch_embedding(x_patches)
        
        # Positional Embedding (dynamic generation/interpolation)
        base_num_patches = self.pos_embedding_base.shape[1]
        base_h = int(base_num_patches**0.5)
        base_w = int(base_num_patches**0.5)

        pos_embedding_2d = self.pos_embedding_base.view(1, base_h, base_w, self.transformer_dim)
        
        interpolated_pos_embedding = F.interpolate(
            pos_embedding_2d.permute(0, 3, 1, 2), # Permute to (N, C, H, W) for F.interpolate
            size=(num_patches_h, num_patches_w),
            mode='bicubic',
            align_corners=False
        ).permute(0, 2, 3, 1) # Permute back to (N, H, W, C)

        interpolated_pos_embedding = rearrange(interpolated_pos_embedding, '1 h w d -> 1 (h w) d')
        x_tokens = x_tokens + interpolated_pos_embedding # Broadcasting takes care of batch dim

        # Transformer
        for block in self.transformer_blocks:
            x_tokens = block(x_tokens)
        
        # Project back to feature map
        x_features = self.patch_to_feature(x_tokens)
        x_features = self.norm_output(x_features)

        # Unpatchify
        output = rearrange(x_features, 'b (h w) (p1 p2 c_out) -> b c_out (h p1) (w p2)',
                           h=num_patches_h, w=num_patches_w, p1=self.patch_size, p2=self.patch_size, c_out=self.out_channels)
        
        return output

# Downsampling Block
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, num_transformer_blocks, transformer_dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.down_conv = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(2)
        self.patch_size = patch_size
        
        self.vit_block = ViT_2D_Block(out_channels, out_channels, patch_size, num_transformer_blocks, transformer_dim, heads, dim_head, mlp_dim)
        self.cbam_module = CBAM(out_channels)
        self.mham_module = MHAM(out_channels)

    def forward(self, x):
        x_conv = self.down_conv(x)
        x_pooled = self.maxpool(x_conv)
        
        _, _, h_pooled, w_pooled = x_pooled.shape

        pad_h = (self.patch_size - (h_pooled % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w_pooled % self.patch_size)) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x_pooled, (0, pad_w, 0, pad_h))
        else:
            x_padded = x_pooled

        x_vit_out_padded = self.vit_block(x_padded)
        
        x_vit_out = x_vit_out_padded[:, :, :h_pooled, :w_pooled]

        x_cbam = self.cbam_module(x_vit_out)
        x_mham = self.mham_module(x_cbam)

        return x_mham, x_vit_out

# Bilinear Polymerization Pooling (BPP)
class BPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # in_channels for conv1x1 should match the input features (fused_features)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, ft, fa): # ft: intrinsic features, fa: attention features
        if ft.shape[2:] != fa.shape[2:]:
            fa = F.interpolate(fa, size=ft.shape[2:], mode='bilinear', align_corners=True)
            
        fused_features = ft * fa
        output = self.relu(self.conv1x1(fused_features))
        return output

# Upsampling Block
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, bilinear=True): # Added skip_channels
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
                conv3x3(in_channels + out_channels, out_channels),
                nn.ReLU(inplace=True),
                conv3x3(out_channels, out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                conv3x3(in_channels // 2 + out_channels, out_channels),
                nn.ReLU(inplace=True),
                conv3x3(out_channels, out_channels),
                nn.ReLU(inplace=True)
            )
        
        # BPP's in_channels must match the channels of the skip features (x2_vit_out, x2_attention_features)
        # BPP's out_channels must match the Up block's out_channels as it will be concatenated with x1
        self.bpp = BPP(skip_channels, out_channels)

    def forward(self, x1, x2_vit_out, x2_attention_features):
        x1 = self.up(x1)
        
        fused_skip_features = self.bpp(x2_vit_out, x2_attention_features)
        
        diffY = fused_skip_features.size()[2] - x1.size()[2]
        diffX = fused_skip_features.size()[3] - x1.size()[3]
        
        if diffX > 0 or diffY > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x1, fused_skip_features], dim=1)
        
        return self.conv(x)

# Final Output Layer
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            conv3x3(in_channels, in_channels),
            nn.ReLU(inplace=True),
            conv3x3(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
    def forward(self, x):
        return self.conv(x)

# Full ViT-UNet Model
class ViT_UNet(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, base_channels=64, patch_size=16, num_transformer_blocks=2, transformer_dim=768, heads=8, dim_head=64, mlp_dim=3072, bilinear=True):
        super().__init__()
        self.in_channels = n_input_channels
        self.num_classes = n_output_channels
        self.bilinear = bilinear
        
        self.inc = Inc(n_input_channels, base_channels)
        
        self.down1 = Down(base_channels, base_channels * 2, patch_size, num_transformer_blocks, transformer_dim, heads, dim_head, mlp_dim)
        self.down2 = Down(base_channels * 2, base_channels * 4, patch_size, num_transformer_blocks, transformer_dim, heads, dim_head, mlp_dim)
        self.down3 = Down(base_channels * 4, base_channels * 8, patch_size, num_transformer_blocks, transformer_dim, heads, dim_head, mlp_dim)
        self.down4 = Down(base_channels * 8, base_channels * 16, patch_size, num_transformer_blocks, transformer_dim, heads, dim_head, mlp_dim)

        factor = 2 if bilinear else 1
        
        # Up blocks initialization with correct in_channels for each stage
        # in_channels for Up block = out_channels of the *previous* decoder stage (or initial deepest encoder stage)
        # out_channels for Up block = desired output channels for *this* decoder stage
        # skip_channels for Up block = channels of the encoder skip connection features
        
        # up1:
        # Input to up1 from encoder (x5_mham from down4) has base_channels * 16 = 1024 channels.
        # Output of up1 should be base_channels * 8 // factor = 256 channels.
        # Skip connections (x4_vit, x4_mham from down3) have base_channels * 8 = 512 channels.
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, base_channels * 8, bilinear) 
        
        # up2:
        # Input to up2 from up1's output has base_channels * 8 // factor = 256 channels.
        # Output of up2 should be base_channels * 4 // factor = 128 channels.
        # Skip connections (x3_vit, x3_mham from down2) have base_channels * 4 = 256 channels.
        self.up2 = Up(base_channels * 8 // factor, base_channels * 4 // factor, base_channels * 4, bilinear)
        
        # up3:
        # Input to up3 from up2's output has base_channels * 4 // factor = 128 channels.
        # Output of up3 should be base_channels * 2 // factor = 64 channels.
        # Skip connections (x2_vit, x2_mham from down1) have base_channels * 2 = 128 channels.
        self.up3 = Up(base_channels * 4 // factor, base_channels * 2 // factor, base_channels * 2, bilinear)
        
        # up4:
        # Input to up4 from up3's output has base_channels * 2 // factor = 64 channels.
        # Output of up4 should be base_channels = 64 channels.
        # Skip connections (x1 from inc) has base_channels = 64 channels.
        self.up4 = Up(base_channels * 2 // factor, base_channels, base_channels, bilinear) 
        
        self.outc = OutConv(base_channels, n_output_channels)

    def forward(self, x):
        original_h, original_w = x.shape[2:]

        x1 = self.inc(x)
        
        x2_mham, x2_vit = self.down1(x1)
        x3_mham, x3_vit = self.down2(x2_mham)
        x4_mham, x4_vit = self.down3(x3_mham)
        x5_mham, x5_vit = self.down4(x4_mham)

        x_dec = self.up1(x5_mham, x4_vit, x4_mham)
        x_dec = self.up2(x_dec, x3_vit, x3_mham)
        x_dec = self.up3(x_dec, x2_vit, x2_mham)
        
        x_dec = self.up4(x_dec, x1, x1) # x1 acts as both skip features for the last UP block
        
        logits = self.outc(x_dec)

        output_h, output_w = logits.shape[2:]
        if output_h > original_h or output_w > original_w:
            diffY = output_h - original_h
            diffX = output_w - original_w
            logits = logits[:, :, diffY // 2 : output_h - (diffY - diffY // 2),
                                   diffX // 2 : output_w - (diffX - diffX // 2)]
        
        return logits
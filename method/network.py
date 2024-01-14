import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers


# Universal utils
def which_act(act_type):
    if act_type == 'relu':
        act = nn.ReLU()
    elif act_type == 'relu6':
        act = nn.ReLU6()
    elif act_type == 'leakyrelu':
        act = nn.LeakyReLU(negative_slope=0.1)
    elif act_type == 'elu':
        act = nn.ELU()
    else:
        raise AttributeError('Unsupported activation_type!')
    return act


# Mid-branch PWConv w. cond
class Cond_Trunk(nn.Module):
    def __init__(self, in_nc=3, nf=32, act_type='leakyrelu'):
        super(Cond_Trunk, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, nf//2, 3, 2, 1)
        self.conv2 = nn.Conv2d(nf//2, nf//2, 3, 2, 1, groups=4)
        self.conv3 = nn.Conv2d(nf//2, nf, 3, 2, 1)
        self.act = which_act(act_type)

    def forward(self, x):
        conv1_out = self.act(self.conv1(x))
        conv2_out = self.act(self.conv2(conv1_out))
        conv3_out = self.act(self.conv3(conv2_out))
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)  # [n, c]
        return out


class PWConv_Cond(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, act_type='leakyrelu'):
        super(PWConv_Cond, self).__init__()

        self.nf = nf
        self.out_nc = out_nc

        self.conv1 = nn.Conv2d(in_nc, nf, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf * 2, 1, 1)
        self.conv3 = nn.Conv2d(nf * 2, nf * 2, 1, 1)
        self.conv4 = nn.Conv2d(nf * 2, nf, 1, 1)
        self.conv5 = nn.Conv2d(nf, out_nc, 1, 1)

        self.act = which_act(act_type)

        self.cond_net = Cond_Trunk(in_nc=in_nc, nf=nf, act_type=act_type)

        self.cond_scale1 = nn.Linear(nf, nf, bias=True)
        self.cond_scale2 = nn.Linear(nf, nf * 2, bias=True)
        self.cond_scale3 = nn.Linear(nf, nf * 2, bias=True)
        self.cond_scale4 = nn.Linear(nf, nf, bias=True)
        self.cond_scale5 = nn.Linear(nf, out_nc, bias=True)

        self.cond_shift1 = nn.Linear(nf, nf, bias=True)
        self.cond_shift2 = nn.Linear(nf, nf * 2, bias=True)
        self.cond_shift3 = nn.Linear(nf, nf * 2, bias=True)
        self.cond_shift4 = nn.Linear(nf, nf, bias=True)
        self.cond_shift5 = nn.Linear(nf, out_nc, bias=True)

    def forward(self, x, prior):

        cond = self.cond_net(prior)

        scale1 = self.cond_scale1(cond)
        scale2 = self.cond_scale2(cond)
        scale3 = self.cond_scale3(cond)
        scale4 = self.cond_scale4(cond)
        scale5 = self.cond_scale5(cond)

        shift1 = self.cond_shift1(cond)
        shift2 = self.cond_shift2(cond)
        shift3 = self.cond_shift3(cond)
        shift4 = self.cond_shift4(cond)
        shift5 = self.cond_shift5(cond)

        out = self.conv1(x)
        out = self.act(out * scale1.view(-1, self.nf, 1, 1) + shift1.view(-1, self.nf, 1, 1) + out)
        out = self.conv2(out)
        out = self.act(out * scale2.view(-1, self.nf * 2, 1, 1) + shift2.view(-1, self.nf * 2, 1, 1) + out)
        out = self.conv3(out)
        out = self.act(out * scale3.view(-1, self.nf * 2, 1, 1) + shift3.view(-1, self.nf * 2, 1, 1) + out)
        out = self.conv4(out)
        out = self.act(out * scale4.view(-1, self.nf, 1, 1) + shift4.view(-1, self.nf, 1, 1) + out)
        out = self.conv5(out)
        out = self.act(out * scale5.view(-1, self.out_nc, 1, 1) + shift5.view(-1, self.out_nc, 1, 1) + out)
        return out


### Below are modules for Restoremer ###
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=24,  # 48
                 num_blocks=None,
                 num_refinement_blocks=4,
                 heads=None,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):

        super(Restormer, self).__init__()

        num_blocks = [1, 1, 1]  # [4, 6, 6, 8]
        heads = [1, 2, 4]  # [1, 2, 4, 8]

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2))  ## From Level 2 to Level 3
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 4))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 4), int(dim * 2), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output = nn.Conv2d(int(dim * 2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        latent = self.latent(inp_enc_level3)

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        return self.output(out_dec_level1)
### End of Restoremer ###


# luminance segmentation function
class BrightIntensityLum(nn.Module):
    def __init__(self, t_b=0.7611):
        super(BrightIntensityLum, self).__init__()
        self.t_b = t_b

    def forward(self, x):
        lum = 0.2627 * x[:, 0, :, :] + 0.6780 * x[:, 1, :, :] + 0.0593 * x[:, 2, :, :]
        lum = torch.clamp(lum / self.t_b, min=0, max=1)
        return lum.unsqueeze(1).repeat(1, 3, 1, 1)


class DarkIntensityLum(nn.Module):
    def __init__(self, t_d=0.2539):
        super(DarkIntensityLum, self).__init__()
        self.t_d = t_d

    def forward(self, x):
        lum = 0.2627 * x[:, 0, :, :] + 0.6780 * x[:, 1, :, :] + 0.0593 * x[:, 2, :, :]
        lum = torch.clamp((lum - 1) / (self.t_d - 1), min=0, max=1)
        return lum.unsqueeze(1).repeat(1, 3, 1, 1)


# Whole network
class TriSegNet(nn.Module):
    def __init__(self, in_nc=3, nf=32, out_nc=3, act_type='leakyrelu'):
        super(TriSegNet, self).__init__()

        self.act = which_act(act_type)

        self.prior_downscale = nn.AvgPool2d(kernel_size=8, stride=8)
        self.mid_branch = PWConv_Cond(in_nc=in_nc, out_nc=out_nc, nf=nf, act_type=act_type)

        self.bright_branch = Restormer(inp_channels=3, out_channels=3, dim=nf // 2, ffn_expansion_factor=2)
        self.dark_branch = Restormer(inp_channels=3, out_channels=3, dim=nf // 2, ffn_expansion_factor=2)

        self.t_b = 0.7611
        self.t_d = 0.7611

        self.bright_intensity = BrightIntensityLum(t_b=self.t_b)
        self.dark_intensity = DarkIntensityLum(t_d=self.t_d)

    def forward(self, x):
        prior = self.prior_downscale(x)

        bright_part = x * self.bright_intensity(x)
        dark_part = x * self.dark_intensity(x)
        mid_part = x

        bright_out = self.bright_branch(bright_part)
        dark_out = self.dark_branch(dark_part)
        mid_out = self.mid_branch(mid_part, prior)

        out = mid_out + bright_out + dark_out

        return out

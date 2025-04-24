import time
import math
from functools import partial
from typing import Optional, Callable
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from scipy.signal import impulse
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import cat

from .S3_DSCNet import EncoderConv
from .S3_DSConv import DSConv

from typing import Optional, Callable, Any
from .S_SS2D import S_SS2D

# import selective_scan_cuda
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop


    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        # self.conv00 = EncoderConv(self.d_inner, self.d_inner)
        # self.conv0x = DSConv(
        #     self.d_inner,
        #     self.d_inner,
        #     kernel_size=d_conv,
        #     extend_scope=1,
        #     morph=0,  # 0 表示沿 x 轴方向的卷积
        #     if_offset=True,
        #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # )
        # self.conv0y = DSConv(
        #     self.d_inner,
        #     self.d_inner,
        #     kernel_size=d_conv,
        #     extend_scope=1,
        #     morph=1,  # 1 表示沿 y 轴方向的卷积
        #     if_offset=True,
        #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # )
        # self.conv1 = EncoderConv(3 * self.d_inner,  self.d_inner)

        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):       # torch.Size([1, 48, 224, 224])
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        # x_00_0 = self.conv00(x)     # torch.Size([1, 96, 224, 224])
        # x_0x_0 = self.conv0x(x)
        # x_0y_0 = self.conv0y(x)
        # enc1 = self.conv1(cat([x_00_0, x_0x_0, x_0y_0], dim=1))  # 在通道维度（dim=1）上拼接在一起。
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        x = self.act(x)  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)      # torch.Size([1, 224, 224, 48])
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    # 定义扩展因子为 1
    expansion = 1
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        # self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)

        self.self_attention = S_SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # bias=False,
            # ==========================
            # dt_min=0.001,
            # dt_max=0.1,
            # dt_init="random",
            # dt_scale="random",
            # dt_init_floor=1e-4,
            initialize=ssm_init,
            # ==========================
            forward_type=forward_type,
        )
        self.drop_path = DropPath(drop_path)

        self.conv00 = EncoderConv(hidden_dim, hidden_dim)
        self.conv0x = DSConv(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            extend_scope=1,
            morph=0,  # 0 表示沿 x 轴方向的卷积
            if_offset=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.conv0y = DSConv(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            extend_scope=1,
            morph=1,  # 1 表示沿 y 轴方向的卷积
            if_offset=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.conv1 = EncoderConv(3 * hidden_dim, hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, input: torch.Tensor):
        input1 = input.permute(0, 2, 3, 1)  # torch.Size([1, 48, 224, 224])
        x_00_0 = self.conv00(input)  # torch.Size([1, 48, 224, 224])
        x_0x_0 = self.conv0x(input)  # torch.Size([1, 48, 224, 224])
        x_0y_0 = self.conv0y(input)
        x1 = self.conv1(cat([x_00_0, x_0x_0, x_0y_0], dim=1))  # 在通道维度（dim=1）上拼接在一起。
        x2 = input + self.drop_path(self.self_attention(self.ln_1(input1)).permute(0, 3, 1, 2)) + x1
        x3 = self.conv2(x2)
        return x3


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x



class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        upsample=None,
        use_checkpoint=False,
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None


    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x



class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[24, 48, 96, 192], dims_decoder=[192, 96, 48, 24], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1]//4, num_classes, 1)
        self.downsample = nn.AvgPool2d(kernel_size=2)  # 使用2x2平均池化将尺寸缩小至1/2

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.feat_size = [24, 48, 96, 192, 384]
        self.hidden_size=192
        self.in_chans = 3
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=2,
            norm_name="batch",  # 使用批归一化
            res_block=True,  # 使用残差块
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name="batch",  # 使用批归一化
            res_block=True,  # 使用残差块
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name="batch",  # 使用批归一化
            res_block=True,  # 使用残差块
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name="batch",  # 使用批归一化
            res_block=True,  # 使用残差块
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name="batch",  # 使用批归一化
            res_block=True,  # 使用残差块
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="batch",  # 使用批归一化
            res_block=True,  # 使用残差块
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="batch",  # 使用批归一化
            res_block=True,  # 使用残差块
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="batch",  # 使用批归一化
            res_block=True,  # 使用残差块
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="batch",  # 使用批归一化
            res_block=True,  # 使用残差块
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name="batch",  # 使用批归一化
            res_block=True,  # 使用残差块
        )

        self.apply(self._init_weights)

        self.n_channels = 3
        self.number = 4
        self.kernel_size = 9
        self.extend_scope = 1.0
        self.if_offset=True
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.up = nn.Upsample(scale_factor=2,
                              mode="bilinear",
                              align_corners=True)

        self.conv120 = EncoderConv(96 * self.number, 32 * self.number)
        self.conv12x = DSConv(
            96 * self.number,
            32 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv12y = DSConv(
            96 * self.number,
            32 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv13 = EncoderConv(96 * self.number, 24 * self.number)

        self.conv140 = DecoderConv(48 * self.number, 16 * self.number)
        self.conv14x = DSConv(
            48 * self.number,
            16 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv14y = DSConv(
            48 * self.number,
            16 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv15 = DecoderConv(48 * self.number, 12 * self.number)

        self.conv160 = DecoderConv(24 * self.number, 8 * self.number)
        self.conv16x = DSConv(
            24 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv16y = DSConv(
            24 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv17 = DecoderConv(24 * self.number, 6 * self.number)

        self.conv180 = DecoderConv(12 * self.number, 4 * self.number)
        self.conv18x = DSConv(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv18y = DSConv(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv19 = DecoderConv(12 * self.number, 6 * self.number)

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv00 = EncoderConv(self.n_channels, self.number)
        self.conv0x = DSConv(
            self.n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,  # 0 表示沿 x 轴方向的卷积
            self.if_offset,
            self.device,
        )
        self.conv0y = DSConv(
            self.n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,  # 1 表示沿 y 轴方向的卷积
            self.if_offset,
            self.device,
        )
        self.conv1 = EncoderConv(3 * self.number, 6 * self.number)

        self.conv20 = EncoderConv(6 * self.number, 6 * self.number)
        self.conv2x = DSConv(
            6 * self.number,
            6 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv2y = DSConv(
            6 * self.number,
            6 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv3 = EncoderConv(18 * self.number, 12 * self.number)

        self.conv40 = EncoderConv(12 * self.number, 12 * self.number)
        self.conv4x = DSConv(
            12 * self.number,
            12 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv4y = DSConv(
            12 * self.number,
            12 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv5 = EncoderConv(36 * self.number, 24 * self.number)

        self.conv60 = EncoderConv(24 * self.number, 24 * self.number)
        self.conv6x = DSConv(
            24 * self.number,
            24 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv6y = DSConv(
            24 * self.number,
            24 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv7 = EncoderConv(72 * self.number, 48 * self.number)
        self.sigmoid = nn.Sigmoid()

        self.dblock1 = MEFFblock(192)
        self.bn1 = normalization(192, norm='gn')
        self.relu1 = nn.ReLU(inplace=True)
        self.dblock2 = MEFFblock(96)
        self.bn2 = normalization(96, norm='gn')
        self.relu2 = nn.ReLU(inplace=True)
        self.dblock3 = MEFFblock(48)
        self.bn3 = normalization(48, norm='gn')
        self.relu3 = nn.ReLU(inplace=True)
        self.dblock4 = MEFFblock(24)
        self.bn4 = normalization(24, norm='gn')
        self.relu4 = nn.ReLU(inplace=True)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x_in):  # torch.Size([1, 3, 224, 224])
        skip_list = []
        x = self.patch_embed(x_in)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)        # torch.Size([1, 56, 56, 96])

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)

        '''for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x+skip_list[-inx])
        '''
        # 将 skip_list 中所有元素从 [B, W, H, C] 转换为 [B, C, W, H]
        skip_list = [tensor.permute(0, 3, 1, 2) for tensor in skip_list]
        x = x.permute(0, 3, 1, 2)   # torch.Size([1, 192, 7, 7])

        # enc1 = self.encoder1(x_in)  # torch.Size([1, 24, 112, 112])
        # 下采样
        x_in = self.downsample(x_in)  # torch.Size([1, 3, 112, 112])
        x_00_0 = self.conv00(x_in)
        x_0x_0 = self.conv0x(x_in)
        x_0y_0 = self.conv0y(x_in)
        enc1 = self.conv1(cat([x_00_0, x_0x_0, x_0y_0], dim=1))  # 在通道维度（dim=1）上拼接在一起。

        x2 = skip_list[0]                                               # torch.Size([1, 24, 56, 56])
        # enc2 = self.encoder2(x2)    # torch.Size([1, 48, 56, 56])
        x_20_0 = self.conv20(x2)
        x_2x_0 = self.conv2x(x2)
        x_2y_0 = self.conv2y(x2)
        enc2 = self.conv3(cat([x_20_0, x_2x_0, x_2y_0], dim=1))

        x3 = skip_list[1]                                               # torch.Size([1, 48, 28, 28])
        # enc3 = self.encoder3(x3)    # torch.Size([1, 96, 28, 28])
        x_40_0 = self.conv40(x3)
        x_4x_0 = self.conv4x(x3)
        x_4y_0 = self.conv4y(x3)
        enc3 = self.conv5(cat([x_40_0, x_4x_0, x_4y_0], dim=1))


        x4 = skip_list[2]                                               # torch.Size([1, 96, 14, 14])
        # enc4 = self.encoder4(x4)    # torch.Size([1, 192, 14, 14])
        x_60_0 = self.conv60(x4)
        x_6x_0 = self.conv6x(x4)
        x_6y_0 = self.conv6y(x4)
        enc4 = self.conv7(cat([x_60_0, x_6x_0, x_6y_0], dim=1))

        enc_hidden = self.encoder5(skip_list[3])#torch.size([1,192,7,7])

        '''
        x1 = self.dblock1(enc4)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        enc4 = enc4 * x1
        x = self.up(enc_hidden)
        x_120_2 = self.conv120(cat([x, enc4], dim=1))
        x_12x_2 = self.conv12x(cat([x, enc4], dim=1))
        x_12y_2 = self.conv12y(cat([x, enc4], dim=1))
        dec3 = self.conv13(cat([x_120_2, x_12x_2, x_12y_2], dim=1))     # torch.Size([1, 96, 14, 14])

        x1 = self.dblock2(enc3)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        enc3 = enc3 * x1
        x = self.up(dec3)
        x_140_2 = self.conv140(cat([x, enc3], dim=1))
        x_14x_2 = self.conv14x(cat([x, enc3], dim=1))
        x_14y_2 = self.conv14y(cat([x, enc3], dim=1))
        dec2 = self.conv15(cat([x_140_2, x_14x_2, x_14y_2], dim=1))     # torch.Size([1, 48, 28, 28])

        x1 = self.dblock3(enc2)
        x1 = self.bn3(x1)
        x1 = self.relu3(x1)
        enc2 = enc2 * x1
        x = self.up(dec2)
        x_160_2 = self.conv160(cat([x, enc2], dim=1))
        x_16x_2 = self.conv16x(cat([x, enc2], dim=1))
        x_16y_2 = self.conv16y(cat([x, enc2], dim=1))
        dec1 = self.conv17(cat([x_160_2, x_16x_2, x_16y_2], dim=1))     # torch.Size([1, 24, 56, 56])

        x1 = self.dblock4(enc1)
        x1 = self.bn4(x1)
        x1 = self.relu4(x1)
        enc1 = enc1 * x1
        x = self.up(dec1)
        x_160_2 = self.conv180(cat([x, enc1], dim=1))
        x_16x_2 = self.conv18x(cat([x, enc1], dim=1))
        x_16y_2 = self.conv18y(cat([x, enc1], dim=1))
        dec0 = self.conv19(cat([x_160_2, x_16x_2, x_16y_2], dim=1))  # torch.Size([1, 24, 112, 112])
        '''

        x1 = self.dblock1(enc4)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        enc4 = enc4 * x1
        dec3 = self.decoder5(enc_hidden, enc4)  # torch.Size([1, 96, 14, 14])

        x1 = self.dblock2(enc3)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        enc3 = enc3 * x1
        dec2 = self.decoder4(dec3, enc3)        # torch.Size([1, 48, 28, 28])

        x1 = self.dblock3(enc2)
        x1 = self.bn3(x1)
        x1 = self.relu3(x1)
        enc2 = enc2 * x1
        dec1 = self.decoder3(dec2, enc2)        # torch.Size([1, 24, 56, 56])

        x1 = self.dblock4(enc1)
        x1 = self.bn4(x1)
        x1 = self.relu4(x1)
        enc1 = enc1 * x1
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)

        out =out.permute(0, 2, 3, 1)
        return out #[1,112,112,96]

    def forward_final(self, x):
        # 将输出上采样至目标大小 [224, 224]
        x = self.final_up(x)        # torch.Size([1, 448, 448, 24])
        x = x.permute(0,3,1,2)
        x = self.final_conv(x)      # torch.Size([1, 1, 448, 448])
        # 下采样至 [112, 112]
        x = self.downsample(x)  # torch.Size([1, 1, 224, 224])
        return x

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_final(x)
        return x #[1,1,224,224]

# MEFF_Block
# 定义一个使用inplace操作的ReLU部分函数
nonlinearity = partial(F.relu, inplace=False)

def normalization(planes, norm='gn'):
    """
    返回一个基于norm类型的归一化层。
    参数:
    - planes (int): 输入张量的通道数。
    - norm (str): 归一化类型 ('bn' 表示BatchNorm, 'gn' 表示GroupNorm, 'in' 表示InstanceNorm)。

    返回:
    - nn.Module: 归一化层。
    """
    if norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)  # 8组的GroupNorm
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    else:
        raise ValueError('不支持的归一化类型 {}'.format(norm))
    return m

class MEFFblock(nn.Module):
    def __init__(self, channel):
        """
        MEFFblock的初始化函数。
        参数:
        - channel (int): 输入和输出张量的通道数。
        """
        super(MEFFblock, self).__init__()
        self.bn1 = normalization(channel, norm='gn')  # 归一化层
        self.relu1 = nn.ReLU(inplace=False)  # ReLU激活函数

        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)  # 空洞卷积，扩张率为1
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)  # 空洞卷积，扩张率为3
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)  # 空洞卷积，扩张率为5

        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)  # 1x1卷积
        '''
        self.conv00 = EncoderConv(channel, channel)
        self.conv0x = DSConv(
            channel,
            channel,
            kernel_size=3,
            extend_scope=1,
            morph=0,  # 0 表示沿 x 轴方向的卷积
            if_offset=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            padding=0
        )
        self.conv0y = DSConv(
            channel,
            channel,
            kernel_size=3,
            extend_scope=1,
            morph=1,  # 1 表示沿 y 轴方向的卷积
            if_offset=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            padding=0
        )
        self.conv1 = EncoderConv(3 * channel, channel)

        self.conv20 = EncoderConv(channel, channel)
        self.conv2x = DSConv(
            channel,
            channel,
            kernel_size=3,
            extend_scope=3,
            morph=0,  # 0 表示沿 x 轴方向的卷积
            if_offset=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            padding=0
        )
        self.conv2y = DSConv(
            channel,
            channel,
            kernel_size=3,
            extend_scope=3,
            morph=1,  # 1 表示沿 y 轴方向的卷积
            if_offset=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            padding=0
        )
        self.conv3 = EncoderConv(3 * channel, channel)

        self.conv30 = EncoderConv(channel, channel)
        self.conv3x = DSConv(
            channel,
            channel,
            kernel_size=3,
            extend_scope=5,
            morph=0,  # 0 表示沿 x 轴方向的卷积
            if_offset=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            padding=0
        )
        self.conv3y = DSConv(
            channel,
            channel,
            kernel_size=3,
            extend_scope=5,
            morph=1,  # 1 表示沿 y 轴方向的卷积
            if_offset=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            padding=0
        )
        self.conv4 = EncoderConv(3 * channel, channel)
        '''
        # 初始化卷积层的偏置
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """
        定义前向传播过程。
        参数:
        - x (Tensor): 输入张量。

        返回:
        - Tensor: 输出张量。gaoshou
        """
        x = self.bn1(x)  # 归一化
        x = self.relu1(x)  # ReLU激活
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate1(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(x)))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # 将所有输出相加
        '''
        x_00_0 = self.conv00(x)     # torch.Size([1, 192, 14, 14])
        x_0x_0 = self.conv0x(x)     # torch.Size([1, 192, 14, 14])
        x_0y_0 = self.conv0y(x)     # torch.Size([1, 192, 14, 14])
        x_0_1 = self.conv1(cat([x_00_0, x_0x_0, x_0y_0], dim=1))
        dilate1_out = nonlinearity(x_0_1)  # 执行第一层空洞卷积并激活
        dilate2_out = nonlinearity(self.conv1x1(x_0_1))  # 执行1x1卷积在第一层空洞卷积输出上并激活

        # block1
        x_20_0 = self.conv20(x)
        x_2x_0 = self.conv2x(x)
        x_2y_0 = self.conv2y(x)
        x_1_1 = self.conv3(cat([x_20_0, x_2x_0, x_2y_0], dim=1))
        dilate3_out = nonlinearity(self.conv1x1(x_1_1))  # 执行1x1卷积在第二层空洞卷积输出上并激活

        # block1
        x_30_0 = self.conv30(x)
        x_3x_0 = self.conv3x(x)
        x_3y_0 = self.conv3y(x)
        x_2_2 = self.conv4(cat([x_30_0, x_3x_0, x_3y_0], dim=1))
        dilate4_out = nonlinearity(self.conv1x1(x_2_2))  # 执行1x1卷积在第三层空洞卷积输出上并激活
        '''
        return out




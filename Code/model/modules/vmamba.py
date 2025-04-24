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
            extend_scope=1.0,
            morph=0,  # 0 表示沿 x 轴方向的卷积
            if_offset=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.conv0y = DSConv(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            extend_scope=1.0,
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


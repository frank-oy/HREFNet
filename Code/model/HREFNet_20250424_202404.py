# --------------------------------------------------------
# High Resolution Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Rao Fu, RainbowSecret
# --------------------------------------------------------

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules.bottleneck_block import Bottleneck

from . import logger
from .modules.vmamba import VSSBlock

blocks_dict = {
    "BOTTLENECK": Bottleneck,
    "VSS_BLOCK": VSSBlock,
}

BN_MOMENTUM = 0.1

class HREFNetModule(nn.Module):
    def __init__(
            self,
            num_branches,
            blocks,
            num_blocks,
            num_inchannels,
            num_channels,
            num_mlp_ratios,
            num_input_resolutions,
            ffn_types,
            multi_scale_output=True,
            drop_paths=0.0,
    ):

        super(HREFNetModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            num_input_resolutions,
            num_mlp_ratios,
            ffn_types,
            drop_paths,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

        self.num_mlp_ratios = num_mlp_ratios
        self.num_input_resolutions = num_input_resolutions
        self.ffn_types = ffn_types

    def _check_branches(
            self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            print(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(
            self,
            branch_index,
            block,
            num_blocks,
            num_channels,
            num_input_resolutions,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            attn_types,
            ffn_types,
            drop_paths,
            stride=1,
    ):
        downsample = None
        if (
                stride != 1
                or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                drop_path=drop_paths[0],
                norm_layer=nn.LayerNorm,
            )
        )


        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion

        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    drop_path=drop_paths[0],
                    norm_layer=nn.LayerNorm,
                )
            )


        return nn.Sequential(*layers)

    def _make_branches(
            self,
            num_branches,
            block,
            num_blocks,
            num_channels,
            num_input_resolutions,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            attn_types,
            ffn_types,
            drop_paths,
    ):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    num_input_resolutions,
                    num_mlp_ratios,
                    attn_types,
                    ffn_types,
                    drop_paths,
                )
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):

        if self.num_branches == 1:
            return None


        num_branches = self.num_branches
        num_inchannels = self.num_inchannels


        fuse_layers = []


        for i in range(num_branches if self.multi_scale_output else 1):

            fuse_layer = []


            for j in range(num_branches):
                if j > i:

                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                kernel_size=1,
                                stride=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_inchannels[i],
                                momentum=BN_MOMENTUM
                            ),
                            nn.Upsample(
                                scale_factor=2 ** (j - i),
                                mode="nearest"
                            ),
                        )
                    )
                elif j == i:

                    fuse_layer.append(None)
                else:

                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:

                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=num_inchannels[j],
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_inchannels[j],
                                        momentum=BN_MOMENTUM
                                    ),
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3,
                                        momentum=BN_MOMENTUM
                                    ),
                                )
                            )
                        else:

                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=num_inchannels[j],
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_inchannels[j],
                                        momentum=BN_MOMENTUM
                                    ),
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3,
                                        momentum=BN_MOMENTUM
                                    ),
                                    nn.ReLU(False),
                                )
                            )


                    fuse_layer.append(nn.Sequential(*conv3x3s))

            fuse_layers.append(nn.ModuleList(fuse_layer))


        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):

        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])


        x_fuse = []

        for i in range(len(self.fuse_layers)):

            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])

            for j in range(1, self.num_branches):
                if i == j:

                    y = y + x[j]
                elif j > i:

                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])

            x_fuse.append(self.relu(y))

        return x_fuse


class HREFNet(nn.Module):
    def __init__(self, cfg,  **kwargs):
        super(HREFNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        depth_s2 = cfg["STAGE2"]["NUM_BLOCKS"][0] * cfg["STAGE2"]["NUM_MODULES"]
        depth_s3 = cfg["STAGE3"]["NUM_BLOCKS"][0] * cfg["STAGE3"]["NUM_MODULES"]
        depth_s4 = cfg["STAGE4"]["NUM_BLOCKS"][0] * cfg["STAGE4"]["NUM_MODULES"]
        depths = [depth_s2, depth_s3, depth_s4]
        drop_path_rate = cfg["DROP_PATH_RATE"]
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, sum(depths))]

        self.stage1_cfg = cfg["STAGE1"]
        num_channels = self.stage1_cfg["NUM_CHANNELS"][0]
        block = blocks_dict[self.stage1_cfg["BLOCK"]]
        num_blocks = self.stage1_cfg["NUM_BLOCKS"][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = cfg["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels
        )
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, drop_paths=dpr[0:depth_s2]
        )

        # 配置Stage 3
        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg,
            num_channels,
            drop_paths=dpr[depth_s2: depth_s2 + depth_s3],
        )

        self.stage4_cfg = cfg["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multi_scale_output=True,
            drop_paths=dpr[depth_s2 + depth_s3:],
        )

        last_inp_channels = int(np.sum(pre_stage_channels))
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=1080,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            torch.nn.SyncBatchNorm(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1)
        )

        self.dblock1 = MREF(256, flag=1)
        self.dblock2 = MREF(96)
        self.dblock3 = MREF(192)
        self.dblock4 = MREF(384)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):

        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):

            if i < num_branches_pre:

                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_channels_cur_layer[i], momentum=BN_MOMENTUM
                            ),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:

                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(
            self,
            block,
            inplanes,
            planes,
            blocks,
            input_resolution=None,
            num_heads=1,
            stride=1,
            window_size=7,
            halo_size=1,
            mlp_ratio=4.0,
            q_dilation=1,
            kv_dilation=1,
            sr_ratio=1,
            attn_type="msw",
    ):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    planes * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []

        layers.append(block(inplanes, planes, stride, downsample))

        inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(
            self, layer_config, num_inchannels, multi_scale_output=True, drop_paths=0.0
    ):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        num_mlp_ratios = layer_config["NUM_MLP_RATIOS"]
        num_input_resolutions = layer_config["NUM_RESOLUTIONS"]
        attn_types = layer_config["ATTN_TYPES"]
        ffn_types = layer_config["FFN_TYPES"]

        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HREFNetModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    num_mlp_ratios,
                    num_input_resolutions,
                    ffn_types[i],
                    reset_multi_scale_output,
                    drop_paths=drop_paths[num_blocks[0] * i: num_blocks[0] * (i + 1)],
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x1 = self.dblock1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        y_list = self.stage2(x_list)
        x2 = self.dblock2(y_list[-1])

        x_list = []

        for i in range(self.stage3_cfg["NUM_BRANCHES"]):

            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))

            else:
                x_list.append(y_list[i])

        y_list = self.stage3(
            x_list)
        x3 = self.dblock3(y_list[-1])   # 192

        x_list = []

        for i in range(self.stage4_cfg["NUM_BRANCHES"]):

            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))

            else:
                x_list.append(y_list[i])

        y_list = self.stage4(x_list)
        x4 = self.dblock4(y_list[-1])
        fusion = x1 * x2 * x3 * x4

        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        x1 = F.interpolate(y_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=None)
        x2 = F.interpolate(y_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=None)
        x3 = F.interpolate(y_list[3], size=(x0_h, x0_w), mode='bilinear', align_corners=None)
        y = torch.cat([y_list[0], x1, x2, x3, fusion],1)

        y = self.last_layer(y)
        y = torch.sigmoid(y)
        return y

    def init_weights(
            self,
            pretrained="",
    ):
        logger.info("=> init weights from normal distribution")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info("=> loading pretrained model {}".format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
            }
            for k, _ in pretrained_dict.items():
                logger.info("=> loading {} pretrained model {}".format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class MREF(nn.Module):
    def __init__(self, channel, norm='gn', flag=0):
        super(MREF, self).__init__()


        self.norm = self.normalization(channel, norm)

        self.activation = nn.ELU(inplace=True)

        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)

        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, padding=0)

        self.conv2561 = nn.Conv2d(256, 360, kernel_size=1)

        self.conv128 = nn.ConvTranspose2d(96, 360, kernel_size=4, stride=2, padding=1)

        self.conv256 = nn.Sequential(
            nn.ConvTranspose2d(192, 360, kernel_size=4, stride=2, padding=1),  # 56→112
            nn.ConvTranspose2d(360, 360, kernel_size=4, stride=2, padding=1)  # 112→224
        )


        self.conv512 = nn.Sequential(
            nn.ConvTranspose2d(384, 360, kernel_size=4, stride=2, padding=1),  # 28→56
            nn.ConvTranspose2d(360, 360, kernel_size=4, stride=2, padding=1),  # 56→112
            nn.ConvTranspose2d(360, 360, kernel_size=4, stride=2, padding=1)  # 112→224
        )

        self.se = SEBlock(360)

        self.flag = flag

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def normalization(channel, norm_type):

        if norm_type == 'bn':
            return nn.BatchNorm2d(channel)
        elif norm_type == 'gn':
            return nn.GroupNorm(32, channel)
        else:
            raise ValueError("Unsupported normalization: {}".format(norm_type))

    def forward(self, x):

        x = self.norm(x)
        x = self.activation(x)

        d1 = self.activation(self.dilate1(x))
        d2 = self.activation(self.conv1x1(self.dilate1(x)))
        d3 = self.activation(self.conv1x1(self.dilate2(x)))
        d4 = self.activation(self.conv1x1(self.dilate3(x)))


        out = x + d1 + d2 + d3 + d4

        if out.shape[1] == 256 and self.flag == 1:
            out = self.conv2561(out)
        elif out.shape[1] == 96:
            out = self.conv128(out)
        elif out.shape[1] == 192:
            out = self.conv256(out)
        elif out.shape[1] == 384:
            out = self.conv512(out)

        out = self.se(out)

        return out

class SEBlock(nn.Module):


    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


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
                    # 如果当前分支索引大于融合层索引，需要插值到当前融合层的分辨率
                    width_output = x[i].shape[-1]  # 当前融合层的宽度
                    height_output = x[i].shape[-2]  # 当前融合层的高度
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),  # 通过相应的融合层处理分支输出
                        size=[height_output, width_output],  # 插值到目标尺寸
                        mode="bilinear",  # 使用双线性插值
                        align_corners=True,  # 对齐角点
                    )
                else:
                    # 如果当前分支索引小于融合层索引，直接通过相应的融合层处理并相加
                    y = y + self.fuse_layers[i][j](x[j])

            # 将融合后的输出通过ReLU激活函数，并添加到融合输出列表中
            x_fuse.append(self.relu(y))

        # 返回所有融合层的输出
        return x_fuse


class HREFNet(nn.Module):
    def __init__(self, cfg,  **kwargs):
        super(HREFNet, self).__init__()

        # 初始化第一层卷积层和批量归一化层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)  # 输入3通道（RGB图像），输出64通道，卷积核大小3x3，步长为2，填充为1
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)  # 对64通道进行批量归一化
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)  # 第二个卷积层，输入64通道，输出64通道，卷积核大小3x3，步长为2，填充为1
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)  # 对64通道进行批量归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数

        # 计算每个阶段的深度，用于随机深度
        depth_s2 = cfg["STAGE2"]["NUM_BLOCKS"][0] * cfg["STAGE2"]["NUM_MODULES"]  # Stage 2 的总深度
        depth_s3 = cfg["STAGE3"]["NUM_BLOCKS"][0] * cfg["STAGE3"]["NUM_MODULES"]  # Stage 3 的总深度
        depth_s4 = cfg["STAGE4"]["NUM_BLOCKS"][0] * cfg["STAGE4"]["NUM_MODULES"]  # Stage 4 的总深度
        depths = [depth_s2, depth_s3, depth_s4]  # 所有阶段的深度列表
        drop_path_rate = cfg["DROP_PATH_RATE"]  # 随机深度的丢弃率
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, sum(depths))]  # 生成从0到drop_path_rate的等间距数值列表，总长度为所有深度的和

        # 配置Stage 1
        self.stage1_cfg = cfg["STAGE1"]
        num_channels = self.stage1_cfg["NUM_CHANNELS"][0]  # Stage 1 的通道数
        block = blocks_dict[self.stage1_cfg["BLOCK"]]  # Stage 1 的块类型
        num_blocks = self.stage1_cfg["NUM_BLOCKS"][0]  # Stage 1 的块数量
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)  # 创建Stage 1
        stage1_out_channel = block.expansion * num_channels  # Stage 1 的输出通道数

        # 配置Stage 2
        self.stage2_cfg = cfg["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]  # Stage 2 的通道数
        block = blocks_dict[self.stage2_cfg["BLOCK"]]  # Stage 2 的块类型
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]  # 更新Stage 2 的通道数
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels
        )  # 创建Stage 1 到Stage 2 的过渡层
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, drop_paths=dpr[0:depth_s2]
        )  # 创建Stage 2

        # 配置Stage 3
        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]  # Stage 3 的通道数
        block = blocks_dict[self.stage3_cfg["BLOCK"]]  # Stage 3 的块类型
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]  # 更新Stage 3 的通道数
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)  # 创建Stage 2 到Stage 3 的过渡层
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg,
            num_channels,
            drop_paths=dpr[depth_s2: depth_s2 + depth_s3],
        )  # 创建Stage 3

        # 配置Stage 4
        self.stage4_cfg = cfg["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]  # Stage 4 的通道数
        block = blocks_dict[self.stage4_cfg["BLOCK"]]  # Stage 4 的块类型
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]  # 更新Stage 4 的通道数
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)  # 创建Stage 3 到Stage 4 的过渡层
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multi_scale_output=True,
            drop_paths=dpr[depth_s2 + depth_s3:],
        )  # 创建Stage 4

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
        # 获取当前层的分支数
        num_branches_cur = len(num_channels_cur_layer)
        # 获取之前层的分支数
        num_branches_pre = len(num_channels_pre_layer)

        # 初始化转换层列表
        transition_layers = []

        # 遍历当前层的每个分支
        for i in range(num_branches_cur):
            # 如果当前分支数小于等于之前的分支数
            if i < num_branches_pre:
                # 检查当前分支和之前分支的通道数是否相同
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    # 如果不同，则添加一个序列，包括卷积层、批归一化和ReLU激活
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],  # 输入通道数
                                num_channels_cur_layer[i],  # 输出通道数
                                3,  # 卷积核大小
                                1,  # 步长
                                1,  # 填充
                                bias=False,  # 不使用偏置
                            ),
                            nn.BatchNorm2d(
                                num_channels_cur_layer[i], momentum=BN_MOMENTUM
                            ),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    # 如果相同，则不需要转换层
                    transition_layers.append(None)
            else:
                # 当前分支数大于之前分支数，需要增加新的卷积层来匹配分支数
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    # 输入通道数为之前最后一个分支的通道数
                    inchannels = num_channels_pre_layer[-1]
                    # 输出通道数为当前分支的通道数（最后一次卷积）或继续保持输入通道数
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    # 添加一个序列，包括卷积层、批归一化和ReLU激活
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                # 将所有卷积序列添加到转换层列表中
                transition_layers.append(nn.Sequential(*conv3x3s))

        # 将转换层列表转换为ModuleList，以便在网络中使用
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
                    inplanes,  # 输入通道数
                    planes * block.expansion,  # 输出通道数
                    kernel_size=1,  # 1x1 卷积
                    stride=stride,  # 步幅
                    bias=False,  # 不使用偏置
                ),
                nn.BatchNorm2d(
                    planes * block.expansion,  # 批归一化层，通道数
                    momentum=BN_MOMENTUM  # 批归一化动量
                ),
            )

        # 初始化层列表
        layers = []

        # 否则，添加一个常规块（如 ResNet 块），并传递降采样层
        layers.append(block(inplanes, planes, stride, downsample))

        # 更新 inplanes 为当前块的输出通道数
        inplanes = planes * block.expansion

        # 添加剩余的 blocks 数量的块
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        # 将层列表打包成 nn.Sequential 并返回
        return nn.Sequential(*layers)

    def _make_stage(
            self, layer_config, num_inchannels, multi_scale_output=True, drop_paths=0.0
    ):
        # 从配置字典中获取各个参数
        num_modules = layer_config["NUM_MODULES"]  # 模块数量
        num_branches = layer_config["NUM_BRANCHES"]  # 分支数量
        num_blocks = layer_config["NUM_BLOCKS"]  # 每个分支中的块数量
        num_channels = layer_config["NUM_CHANNELS"]  # 每个分支的通道数
        block = blocks_dict[layer_config["BLOCK"]]  # 块类型（比如残差块）
        num_mlp_ratios = layer_config["NUM_MLP_RATIOS"]  # 每个分支的MLP比率
        num_input_resolutions = layer_config["NUM_RESOLUTIONS"]  # 每个分支的输入分辨率数量
        attn_types = layer_config["ATTN_TYPES"]  # 每个模块的注意力类型
        ffn_types = layer_config["FFN_TYPES"]  # 每个模块的前馈神经网络类型

        # 初始化模块列表
        modules = []
        for i in range(num_modules):
            # multi_scale_output 仅在最后一个模块使用
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            # 创建高分辨率Transformer模块并添加到模块列表中
            modules.append(
                HREFNetModule(
                    num_branches,  # 分支数量
                    block,  # 块类型
                    num_blocks,  # 每个分支的块数量
                    num_inchannels,  # 输入通道数
                    num_channels,  # 每个分支的通道数
                    num_mlp_ratios,  # 每个分支的MLP比率
                    num_input_resolutions,  # 每个分支的输入分辨率数量
                    ffn_types[i],  # 当前模块的前馈神经网络类型
                    reset_multi_scale_output,  # 是否重置多尺度输出
                    drop_paths=drop_paths[num_blocks[0] * i: num_blocks[0] * (i + 1)],  # 当前模块的drop path
                )
            )
            # 更新输入通道数为当前模块的输出通道数
            num_inchannels = modules[-1].get_num_inchannels()

        # 将所有模块封装为一个顺序容器，并返回该容器和更新后的输入通道数
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):  # torch.Size([1, 1, 224, 224])
        # 对输入进行第一个卷积操作
        x = self.conv1(x)  # torch.Size([1, 64, 224, 224])
        # 对卷积结果进行批归一化
        x = self.bn1(x)
        # 通过ReLU激活函数
        x = self.relu(x)

        # 对输入进行第二个卷积操作
        x = self.conv2(x)  # torch.Size([1, 64, 224, 224])
        # 对卷积结果进行批归一化
        x = self.bn2(x)
        # 再次通过ReLU激活函数
        x = self.relu(x)

        # 进入第一个残差层（ResNet Layer 1）
        x = self.layer1(x)  # torch.Size([1, 256, 224, 224])
        x1 = self.dblock1(x)    # 256

        # 初始化存储不同分支输出的列表
        x_list = []
        # 遍历第二阶段的每个分支
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            # 如果存在相应的转换层，则通过该层转换后加入列表
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            # 如果没有转换层，直接加入输入
            else:
                x_list.append(x)
        # 将这些分支的输出传入第二阶段网络
        y_list = self.stage2(x_list)  # torch.Size([1, 48, 224, 224])    torch.Size([1, 96, 112, 112])
        x2 = self.dblock2(y_list[-1])  # 96

        # 初始化存储不同分支输出的列表
        x_list = []
        # 遍历第三阶段的每个分支
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            # 如果存在相应的转换层，则通过该层转换后加入列表
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            # 如果没有转换层，直接加入对应的第二阶段输出
            else:
                x_list.append(y_list[i])
        # 将这些分支的输出传入第三阶段网络
        y_list = self.stage3(
            x_list)  # torch.Size([1, 48, 224, 224])   torch.Size([1, 96, 112, 112])    torch.Size([1, 192, 56, 56])
        x3 = self.dblock3(y_list[-1])   # 192

        # 初始化存储不同分支输出的列表
        x_list = []
        # 遍历第四阶段的每个分支
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            # 如果存在相应的转换层，则通过该层转换后加入列表
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            # 如果没有转换层，直接加入对应的第三阶段输出
            else:
                x_list.append(y_list[i])
        # 将这些分支的输出传入第四阶段网络
        y_list = self.stage4(x_list)  # torch.Size([1, 48, 224, 224])   torch.Size([1, 96, 112, 112])
        # torch.Size([1, 192, 56, 56])    torch.Size([1, 384, 28, 28])
        x4 = self.dblock4(y_list[-1])   # 384
        fusion = x1 * x2 * x3 * x4

        # Upsampling
        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        x1 = F.interpolate(y_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=None)  # torch.Size([1, 96, 224, 224])
        x2 = F.interpolate(y_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=None)  # torch.Size([1, 192, 224, 224])
        x3 = F.interpolate(y_list[3], size=(x0_h, x0_w), mode='bilinear', align_corners=None)  # torch.Size([1, 384, 224, 224])
        y = torch.cat([y_list[0], x1, x2, x3, fusion],1)  # torch.Size([1, 720, 224, 224])         [1, 1080, 224, 224]

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

        # 归一化层
        self.norm = self.normalization(channel, norm)

        # 激活函数（ELU 更适合边缘细节）
        self.activation = nn.ELU(inplace=True)

        # 多尺度膨胀卷积
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)

        # 1×1 卷积融合
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, padding=0)

        # 处理 256 通道（无需上采样）
        self.conv2561 = nn.Conv2d(256, 360, kernel_size=1)

        # 处理 128 通道（112→224）
        self.conv128 = nn.ConvTranspose2d(96, 360, kernel_size=4, stride=2, padding=1)

        # 处理 256 通道（56→224）
        self.conv256 = nn.Sequential(
            nn.ConvTranspose2d(192, 360, kernel_size=4, stride=2, padding=1),  # 56→112
            nn.ConvTranspose2d(360, 360, kernel_size=4, stride=2, padding=1)  # 112→224
        )

        # 处理 512 通道（28→224）
        self.conv512 = nn.Sequential(
            nn.ConvTranspose2d(384, 360, kernel_size=4, stride=2, padding=1),  # 28→56
            nn.ConvTranspose2d(360, 360, kernel_size=4, stride=2, padding=1),  # 56→112
            nn.ConvTranspose2d(360, 360, kernel_size=4, stride=2, padding=1)  # 112→224
        )

        # 加入通道注意力机制 (SE Block)
        self.se = SEBlock(360)

        self.flag = flag

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def normalization(channel, norm_type):
        """归一化选择"""
        if norm_type == 'bn':
            return nn.BatchNorm2d(channel)
        elif norm_type == 'gn':
            return nn.GroupNorm(32, channel)
        else:
            raise ValueError("Unsupported normalization: {}".format(norm_type))

    def forward(self, x):
        """前向传播"""
        x = self.norm(x)
        x = self.activation(x)

        # 多尺度特征提取
        d1 = self.activation(self.dilate1(x))  # 膨胀1
        d2 = self.activation(self.conv1x1(self.dilate1(x)))
        d3 = self.activation(self.conv1x1(self.dilate2(x)))  # 膨胀3 + 1x1融合
        d4 = self.activation(self.conv1x1(self.dilate3(x)))  # 膨胀5 + 1x1融合

        # 特征融合 + 残差连接
        out = x + d1 + d2 + d3 + d4

        if out.shape[1] == 256 and self.flag == 1:
            out = self.conv2561(out)
        elif out.shape[1] == 96:
            out = self.conv128(out)
        elif out.shape[1] == 192:
            out = self.conv256(out)
        elif out.shape[1] == 384:
            out = self.conv512(out)

        # 通过注意力机制增强重要特征
        out = self.se(out)

        return out

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 注意力机制 (SE Block)
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)  # 全局平均池化
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y  # 通道注意力加权


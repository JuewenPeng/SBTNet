import torch
import torch.nn as nn
import torch.nn.functional as F
from CrossFormer import CrossFormerBlock


def conv(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True, act='LeakyReLU'):
    if act is not None:
        if act == 'LeakyReLU':
            act_ = nn.LeakyReLU(0.1, inplace=True)
        elif act == 'Sigmoid':
            act_ = nn.Sigmoid()
        elif act == 'Tanh':
            act_ = nn.Tanh()

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) // 2) * dilation, bias=bias),
            act_
        )
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                         padding=((kernel_size - 1) // 2) * dilation, bias=bias)


def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


def resnet_block(in_channels, kernel_size=3, dilation=[1, 1], bias=True, res_num=1):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias, res_num=res_num)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias, res_num):
        super(ResnetBlock, self).__init__()
        self.res_num = res_num
        self.stem = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0],
                          padding=((kernel_size - 1) // 2) * dilation[0], bias=bias),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1],
                          padding=((kernel_size - 1) // 2) * dilation[1], bias=bias),
            ) for i in range(res_num)
        ])

    def forward(self, x):

        if self.res_num > 1:
            temp = x

        for i in range(self.res_num):
            xx = self.stem[i](x)
            x = x + xx
        if self.res_num > 1:
            x = x + temp

        return x


def FAC(feat_in, kernel, ksize):
    """
    customized FAC
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    if channels == 3 and kernels == ksize * ksize:
        ####
        kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, 1, ksize, ksize)
        kernel = torch.cat([kernel, kernel, kernel], channels)
        kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)

    else:
        kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)
        kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)

    feat_out = torch.sum(feat_in * kernel, -1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()

    return feat_out


class DRBNet_single(nn.Module):
    def __init__(self, ):
        super(DRBNet_single, self).__init__()

        ks = 3

        ch1 = 32
        ch2 = ch1 * 2
        ch3 = ch1 * 4
        ch4 = ch1 * 8
        self.ch4 = ch4
        self.kernel_width = 7
        self.kernel_dim = self.kernel_width * self.kernel_width

        # feature extractor
        self.conv1_1 = conv(3, ch1, kernel_size=ks, stride=1)
        self.conv1_2 = conv(ch1, ch1, kernel_size=ks, stride=1)
        self.conv1_3 = conv(ch1, ch1, kernel_size=ks, stride=1)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.conv2_2 = conv(ch2, ch2, kernel_size=ks, stride=1)
        self.conv2_3 = conv(ch2, ch2, kernel_size=ks, stride=1)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.conv3_2 = conv(ch3, ch3, kernel_size=ks, stride=1)
        self.conv3_3 = conv(ch3, ch3, kernel_size=ks, stride=1)

        self.conv4_1 = conv(ch3, ch4, kernel_size=ks, stride=2)
        self.conv4_2 = conv(ch4, ch4, kernel_size=ks, stride=1)
        self.conv4_3 = conv(ch4, ch4, kernel_size=ks, stride=1)

        self.conv4_4 = nn.Sequential(
            conv(ch4, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=1),
            resnet_block(ch4, kernel_size=ks, res_num=1),
            conv(ch4, ch4, kernel_size=ks))

        self.upconv3_u = upconv(ch4, ch3)
        self.upconv3_1 = resnet_block(ch3, kernel_size=ks, res_num=1)
        self.upconv3_2 = resnet_block(ch3, kernel_size=ks, res_num=1)
        # here has a dynamic filter and res

        self.img_d8_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch4, kernel_size=ks, stride=1)
        )

        self.upconv3_kernel = nn.Sequential(
            conv(ch4 * 2, ch4, kernel_size=ks, stride=1),
            conv(ch4, ch3, kernel_size=ks, stride=1),
            conv(ch3, self.kernel_dim, kernel_size=1, stride=1, act=None)
        )

        self.upconv3_res = nn.Sequential(
            conv(ch4 * 2, ch4, kernel_size=ks, stride=1),
            conv(ch4, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)
        )

        self.upconv2_u = upconv(ch3, ch2)
        self.upconv2_1 = resnet_block(ch2, kernel_size=ks, res_num=1)
        self.upconv2_2 = resnet_block(ch2, kernel_size=ks, res_num=1)

        self.img_d4_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch3, kernel_size=ks, stride=1),
        )

        self.upconv2_kernel = nn.Sequential(
            conv(ch3 * 2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch3, kernel_size=ks, stride=1),
            conv(ch3, self.kernel_dim, kernel_size=1, stride=1, act=None)
        )

        self.upconv2_res = nn.Sequential(
            conv(ch3 * 2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)
        )
        self.img_d2_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1)
        )

        self.upconv1_u = upconv(ch2, ch1)
        self.upconv1_1 = resnet_block(ch1, kernel_size=ks, res_num=1)
        self.upconv1_2 = resnet_block(ch1, kernel_size=ks, res_num=1)

        self.img_d1_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch1, kernel_size=ks, stride=1),
        )

        self.upconv1_kernel = nn.Sequential(
            conv(ch2 * 2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, self.kernel_dim, kernel_size=1, stride=1, act=None)
        )

        self.upconv1_res = nn.Sequential(
            conv(ch2 * 2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)
        )

        self.upconv0_kernel = nn.Sequential(
            conv(ch1 * 2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, self.kernel_dim, kernel_size=1, stride=1, act=None)
        )

        self.upconv0_res = nn.Sequential(
            conv(ch1 * 2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)
        )

    ##########################################################################
    def forward(self, C):
        # feature extractor
        f1 = self.conv1_3(self.conv1_2(self.conv1_1(C)))
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(f1)))
        f3 = self.conv3_3(self.conv3_2(self.conv3_1(f2)))
        f_C = self.conv4_3(self.conv4_2(self.conv4_1(f3)))

        f = self.conv4_4(f_C)

        img_d8 = F.interpolate(C[:, :3], scale_factor=1 / 8, mode='area')
        img_d8_feature = self.img_d8_feature(img_d8)
        feature_d8 = torch.cat([f, img_d8_feature], 1)  # ch4*2
        kernel_d8 = self.upconv3_kernel(feature_d8)

        res_f8 = self.upconv3_res(feature_d8)

        est_img_d8 = img_d8 + FAC(img_d8, kernel_d8, self.kernel_width) + res_f8

        f = self.upconv3_u(f) + f3
        f = self.upconv3_2(self.upconv3_1(f))

        est_img_d4_interpolate = F.interpolate(est_img_d8, scale_factor=2, mode='area')

        img_d4_feature = self.img_d4_feature(est_img_d4_interpolate)
        feature_d4 = torch.cat([f, img_d4_feature], 1)
        kernel_d4 = self.upconv2_kernel(feature_d4)

        res_f4 = self.upconv2_res(feature_d4)

        est_img_d4 = est_img_d4_interpolate + FAC(est_img_d4_interpolate, kernel_d4, self.kernel_width) + res_f4

        f = self.upconv2_u(f) + f2
        f = self.upconv2_2(self.upconv2_1(f))

        est_img_d2_interpolate = F.interpolate(est_img_d4, scale_factor=2, mode='area')

        img_d2_feature = self.img_d2_feature(est_img_d2_interpolate)
        feature_d2 = torch.cat([f, img_d2_feature], 1)

        kernel_d2 = self.upconv1_kernel(feature_d2)
        res_f2 = self.upconv1_res(feature_d2)

        est_img_d2 = est_img_d2_interpolate + FAC(est_img_d2_interpolate, kernel_d2, self.kernel_width) + res_f2

        f = self.upconv1_u(f) + f1
        f = self.upconv1_2(self.upconv1_1(f))

        est_img_d1_interploate = F.interpolate(est_img_d2, scale_factor=2, mode='area')

        img_d1_feature = self.img_d1_feature(est_img_d1_interploate)
        feature_d1 = torch.cat([f, img_d1_feature], 1)
        kernel_d1 = self.upconv0_kernel(feature_d1)

        res_f1 = self.upconv0_res(feature_d1)

        est_img_d1 = est_img_d1_interploate + FAC(est_img_d1_interploate, kernel_d1, self.kernel_width) + res_f1

        est_img_d1_ = torch.clip(est_img_d1, -1.0, 1.0)

        return est_img_d1_


class AlphaNet(nn.Module):
    def __init__(self, ):
        super().__init__()

        ks = 3

        ch1 = 64
        ch2 = ch1 * 2
        ch3 = ch1 * 4
        ch4 = ch1 * 8
        ch5 = ch1 * 8

        # feature extractor
        self.conv1_1 = conv(3, ch1, kernel_size=ks, stride=1)
        self.conv1_2 = conv(ch1, ch1, kernel_size=ks, stride=1)
        self.conv1_3 = conv(ch1, ch1, kernel_size=ks, stride=1)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.conv2_2 = conv(ch2, ch2, kernel_size=ks, stride=1)
        self.conv2_3 = conv(ch2, ch2, kernel_size=ks, stride=1)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.conv3_2 = conv(ch3, ch3, kernel_size=ks, stride=1)
        self.conv3_3 = conv(ch3, ch3, kernel_size=ks, stride=1)

        self.conv4_1 = conv(ch3, ch4, kernel_size=ks, stride=2)
        self.conv4_2 = conv(ch4, ch4, kernel_size=ks, stride=1)
        self.conv4_3 = conv(ch4, ch4, kernel_size=ks, stride=1)

        ######################## TODO: comment
        # self.conv5_1 = conv(ch4, ch5, kernel_size=ks, stride=2)
        # self.conv5_2 = conv(ch5, ch5, kernel_size=ks, stride=1)
        # self.conv5_3 = conv(ch5, ch5, kernel_size=ks, stride=1)
        #
        # self.conv5_4 = nn.Sequential(
        #     conv(ch5, ch5, kernel_size=ks),
        #     resnet_block(ch5, kernel_size=ks, res_num=1),
        #     resnet_block(ch5, kernel_size=ks, res_num=1),
        #     conv(ch5, ch5, kernel_size=ks))
        #
        # self.upconv4_u = upconv(ch5, ch4)
        # self.upconv4_0 = conv(ch4, ch4, kernel_size=ks, stride=1)
        # self.upconv4_1 = resnet_block(ch4, kernel_size=ks, res_num=1)
        # self.upconv4_2 = resnet_block(ch4, kernel_size=ks, res_num=1)
        #########################

        self.conv4_4 = nn.Sequential(
            conv(ch4, ch4, kernel_size=ks),
            CrossFormerBlock(ch4, (1000, 1000), num_heads=8, group_size=6, lsda_flag=0),
            CrossFormerBlock(ch4, (1000, 1000), num_heads=8, group_size=6, lsda_flag=1),
            CrossFormerBlock(ch4, (1000, 1000), num_heads=8, group_size=6, lsda_flag=0),
            CrossFormerBlock(ch4, (1000, 1000), num_heads=8, group_size=6, lsda_flag=1),
            CrossFormerBlock(ch4, (1000, 1000), num_heads=8, group_size=6, lsda_flag=0),
            CrossFormerBlock(ch4, (1000, 1000), num_heads=8, group_size=6, lsda_flag=1),
            CrossFormerBlock(ch4, (1000, 1000), num_heads=8, group_size=6, lsda_flag=0),
            CrossFormerBlock(ch4, (1000, 1000), num_heads=8, group_size=6, lsda_flag=1),
            conv(ch4, ch4, kernel_size=ks))

        self.upconv3_u = upconv(ch4, ch3)
        self.upconv3_0 = conv(ch3, ch3, kernel_size=ks, stride=1)
        self.upconv3_1 = resnet_block(ch3, kernel_size=ks, res_num=1)
        self.upconv3_2 = resnet_block(ch3, kernel_size=ks, res_num=1)

        self.upconv2_u = upconv(ch3, ch2)
        self.upconv2_0 = conv(ch2, ch2, kernel_size=ks, stride=1)
        self.upconv2_1 = resnet_block(ch2, kernel_size=ks, res_num=1)
        self.upconv2_2 = resnet_block(ch2, kernel_size=ks, res_num=1)

        self.upconv1_u = upconv(ch2, ch1)
        self.upconv1_0 = conv(ch1, ch1, kernel_size=ks, stride=1)
        self.upconv1_1 = resnet_block(ch1, kernel_size=ks, res_num=1)
        self.upconv1_2 = resnet_block(ch1, kernel_size=ks, res_num=1)

        self.out_res = conv(ch1, 1, kernel_size=3, act='Sigmoid')

    ##########################################################################
    def forward(self, C):
        # feature extractor
        f1 = self.conv1_3(self.conv1_2(self.conv1_1(C)))
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(f1)))
        f3 = self.conv3_3(self.conv3_2(self.conv3_1(f2)))
        f4 = self.conv4_3(self.conv4_2(self.conv4_1(f3)))
        # f_C = self.conv5_3(self.conv5_2(self.conv5_1(f4)))

        # f = self.conv5_4(f_C)

        # f = self.upconv4_u(f) + self.upconv4_0(f4)
        # f = self.upconv4_2(self.upconv4_1(f))

        f = self.conv4_4(f4)

        f = self.upconv3_u(f) + self.upconv3_0(f3)
        f = self.upconv3_2(self.upconv3_1(f))

        f = self.upconv2_u(f) + self.upconv2_0(f2)
        f = self.upconv2_2(self.upconv2_1(f))

        f = self.upconv1_u(f) + self.upconv1_0(f1)
        f = self.upconv1_2(self.upconv1_1(f))

        alpha = self.out_res(f)

        return alpha


class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        ks = 3

        ch1 = 32
        ch2 = ch1 * 2
        ch3 = ch1 * 4
        ch4 = ch1 * 8

        # feature extractor
        self.conv1_1 = conv(7, ch1, kernel_size=ks, stride=1)
        self.conv1_2 = conv(ch1, ch1, kernel_size=ks, stride=1)
        self.conv1_3 = conv(ch1, ch1, kernel_size=ks, stride=1)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.conv2_2 = conv(ch2, ch2, kernel_size=ks, stride=1)
        self.conv2_3 = conv(ch2, ch2, kernel_size=ks, stride=1)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.conv3_2 = conv(ch3, ch3, kernel_size=ks, stride=1)
        self.conv3_3 = conv(ch3, ch3, kernel_size=ks, stride=1)

        self.conv4_1 = conv(ch3, ch4, kernel_size=ks, stride=2)
        self.conv4_2 = conv(ch4, ch4, kernel_size=ks, stride=1)
        self.conv4_3 = conv(ch4, ch4, kernel_size=ks, stride=1)

        self.conv4_4 = nn.Sequential(
            conv(ch4, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=1),
            resnet_block(ch4, kernel_size=ks, res_num=1),
            conv(ch4, ch4, kernel_size=ks))

        self.upconv3_u = upconv(ch4, ch3)
        self.upconv3_0 = conv(ch3, ch3, kernel_size=ks, stride=1)
        self.upconv3_1 = resnet_block(ch3, kernel_size=ks, res_num=1)
        self.upconv3_2 = resnet_block(ch3, kernel_size=ks, res_num=1)

        self.upconv2_u = upconv(ch3, ch2)
        self.upconv2_0 = conv(ch2, ch2, kernel_size=ks, stride=1)
        self.upconv2_1 = resnet_block(ch2, kernel_size=ks, res_num=1)
        self.upconv2_2 = resnet_block(ch2, kernel_size=ks, res_num=1)

        self.upconv1_u = upconv(ch2, ch1)
        self.upconv1_0 = conv(ch1, ch1, kernel_size=ks, stride=1)
        self.upconv1_1 = resnet_block(ch1, kernel_size=ks, res_num=1)
        self.upconv1_2 = resnet_block(ch1, kernel_size=ks, res_num=1)

        self.out_res = conv(ch1, 3, kernel_size=3, act=None)

        # self.downsample = nn.PixelUnshuffle(2)
        # self.layer = nn.Sequential(
        #     conv(12 + 3, ch, kernel_size=ks, stride=1),
        #     resnet_block(ch, kernel_size=ks, res_num=1),
        #     resnet_block(ch, kernel_size=ks, res_num=1),
        #     resnet_block(ch, kernel_size=ks, res_num=1),
        #     resnet_block(ch, kernel_size=ks, res_num=1),
        #     nn.Conv2d(ch, 12, kernel_size=ks, stride=1, padding=(ks - 1) // 2)
        # )
        # self.upsample = nn.PixelShuffle(2)

    def forward(self, x_lr, alpha_lr, src):  # , cateye_coord):
        x = F.interpolate(x_lr, size=src.shape[2:], mode='bilinear', align_corners=True)
        alpha = F.interpolate(alpha_lr, size=src.shape[2:], mode='bilinear', align_corners=True)
        # x = torch.cat([x, alpha, src, cateye_coord], dim=1)
        x = torch.cat([x, alpha, src], dim=1)

        f1 = self.conv1_3(self.conv1_2(self.conv1_1(x)))
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(f1)))
        f3 = self.conv3_3(self.conv3_2(self.conv3_1(f2)))
        f4 = self.conv4_3(self.conv4_2(self.conv4_1(f3)))

        f = self.conv4_4(f4)

        f = self.upconv3_u(f) + f3
        f = self.upconv3_2(self.upconv3_1(f))

        f = self.upconv2_u(f) + f2
        f = self.upconv2_2(self.upconv2_1(f))

        f = self.upconv1_u(f) + f1
        f = self.upconv1_2(self.upconv1_1(f))

        x = self.out_res(f)
        return x  # torch.clip(x, -1.0, 1.0)


import copy
class DRBNet_multi(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        network = DRBNet_single()
        ckpt_path = '/data4/pengjuewen/Code/NTIRE23BokehTransformation/models/single_drbnet.pth'
        if pretrained:
            network.load_state_dict(torch.load(ckpt_path))
        old_conv1_1_0 = network.conv1_1[0]
        new_conv1_1_0 = nn.Conv2d(
            in_channels=old_conv1_1_0.in_channels + 4,
            out_channels=old_conv1_1_0.out_channels,
            kernel_size=old_conv1_1_0.kernel_size,
            stride=old_conv1_1_0.stride,
            padding=old_conv1_1_0.padding,
            bias=True if old_conv1_1_0.bias is not None else False,
        )
        if pretrained:
            new_conv1_1_0.weight[:, :3, :, :].data.copy_(old_conv1_1_0.weight.clone())
        network.conv1_1[0] = new_conv1_1_0
        self.net14to18 = copy.deepcopy(network)
        self.net14to160 = copy.deepcopy(network)
        self.net18to14 = copy.deepcopy(network)
        self.net18to18 = copy.deepcopy(network)
        self.net18to160 = copy.deepcopy(network)
        self.net160to14 = copy.deepcopy(network)
        self.net160to18 = copy.deepcopy(network)

        self.alphanet = AlphaNet()

    def forward(self, src, src_lens_type, tgt_lens_type, src_F, tgt_F, disparity):
        x = src * 2 - 1

        alpha = self.alphanet(x)

        # b, c, h, w = src.shape
        # x = F.interpolate(src, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = torch.cat([
            x,
            alpha,
            torch.ones_like(x[:, :1]) * src_lens_type[:, None, None, None],
            torch.ones_like(x[:, :1]) * tgt_lens_type[:, None, None, None],
            torch.ones_like(x[:, :1]) * disparity[:, None, None, None]
        ], dim=1)
        if src.shape[0] == 1:
            if src_F[0] == 1.4 and tgt_F[0] == 1.8:
                x = self.net14to18(x)
            elif src_F[0] == 1.4 and tgt_F[0] == 16.0:
                x = self.net14to160(x)
            elif src_F[0] == 1.8 and tgt_F[0] == 1.4:
                x = self.net18to14(x)
            elif src_F[0] == 1.8 and tgt_F[0] == 1.8:
                x = self.net18to18(x)
            elif src_F[0] == 1.8 and tgt_F[0] == 16.0:
                x = self.net18to160(x)
            elif src_F[0] == 16.0 and tgt_F[0] == 1.4:
                x = self.net160to14(x)
            elif src_F[0] == 16.0 and tgt_F[0] == 1.8:
                x = self.net160to18(x)
            else:
                print('F-value error')
                exit()
        else:
            x = ((src_F == 1.4) * (tgt_F == 1.8))[:, None, None, None] * self.net14to18(x) + \
                ((src_F == 1.4) * (tgt_F == 16.0))[:, None, None, None] * self.net14to160(x) + \
                ((src_F == 1.8) * (tgt_F == 1.4))[:, None, None, None] * self.net18to14(x) + \
                ((src_F == 1.8) * (tgt_F == 1.8))[:, None, None, None] * self.net18to18(x) + \
                ((src_F == 1.8) * (tgt_F == 16.0))[:, None, None, None] * self.net18to160(x) + \
                ((src_F == 16.0) * (tgt_F == 1.4))[:, None, None, None] * self.net160to14(x) + \
                ((src_F == 16.0) * (tgt_F == 1.8))[:, None, None, None] * self.net160to18(x)

        # out = self.refinenet(x, src)
        # out = 0.5 * out + 0.5
        # out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        x = 0.5 * x + 0.5
        # x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x, alpha




class DRBFeature(nn.Module):
    def __init__(self, pretrained_path=None):
        super(DRBFeature, self).__init__()

        ks = 3

        ch1 = 64
        ch2 = ch1 * 2
        ch3 = ch1 * 4
        ch4 = ch1 * 8

        # feature extractor
        self.conv1_1 = conv(8, ch1, kernel_size=ks, stride=1)
        self.conv1_2 = conv(ch1, ch1, kernel_size=ks, stride=1)
        self.conv1_3 = conv(ch1, ch1, kernel_size=ks, stride=1)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.conv2_2 = conv(ch2, ch2, kernel_size=ks, stride=1)
        self.conv2_3 = conv(ch2, ch2, kernel_size=ks, stride=1)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.conv3_2 = conv(ch3, ch3, kernel_size=ks, stride=1)
        self.conv3_3 = conv(ch3, ch3, kernel_size=ks, stride=1)

        self.conv4_1 = conv(ch3, ch4, kernel_size=ks, stride=2)
        self.conv4_2 = conv(ch4, ch4, kernel_size=ks, stride=1)
        self.conv4_3 = conv(ch4, ch4, kernel_size=ks, stride=1)

        self.conv4_4 = nn.Sequential(
            conv(ch4, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=1),
            resnet_block(ch4, kernel_size=ks, res_num=1),
            conv(ch4, ch4, kernel_size=ks))

        self.upconv3_u = upconv(ch4, ch3)
        self.upconv3_1 = resnet_block(ch3, kernel_size=ks, res_num=1)
        self.upconv3_2 = resnet_block(ch3, kernel_size=ks, res_num=1)
        # here has a dynamic filter and res

        self.upconv2_u = upconv(ch3, ch2)
        self.upconv2_1 = resnet_block(ch2, kernel_size=ks, res_num=1)
        self.upconv2_2 = resnet_block(ch2, kernel_size=ks, res_num=1)

        self.upconv1_u = upconv(ch2, ch1)
        self.upconv1_1 = resnet_block(ch1, kernel_size=ks, res_num=1)
        self.upconv1_2 = resnet_block(ch1, kernel_size=ks, res_num=1)

        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path), strict=False)

    ##########################################################################
    def forward(self, C):
        fs = []
        # feature extractor
        f1 = self.conv1_3(self.conv1_2(self.conv1_1(C)))
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(f1)))
        f3 = self.conv3_3(self.conv3_2(self.conv3_1(f2)))
        f_C = self.conv4_3(self.conv4_2(self.conv4_1(f3)))

        f = self.conv4_4(f_C)
        fs.append(f)

        f = self.upconv3_u(f) + f3
        f = self.upconv3_2(self.upconv3_1(f))
        fs.append(f)

        f = self.upconv2_u(f) + f2
        f = self.upconv2_2(self.upconv2_1(f))
        fs.append(f)

        f = self.upconv1_u(f) + f1
        f = self.upconv1_2(self.upconv1_1(f))
        fs.append(f)

        return fs


class DRBKernel(nn.Module):
    def __init__(self, pretrained_path=None):
        super(DRBKernel, self).__init__()

        ks = 3

        ch1 = 64
        ch2 = ch1 * 2
        ch3 = ch1 * 4
        ch4 = ch1 * 8
        self.kernel_width = 7
        self.kernel_dim = self.kernel_width * self.kernel_width

        self.img_d8_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch4, kernel_size=ks, stride=1)
        )

        self.upconv3_kernel = nn.Sequential(
            conv(ch4 * 2, ch4, kernel_size=ks, stride=1),
            conv(ch4, ch3, kernel_size=ks, stride=1),
            conv(ch3, self.kernel_dim, kernel_size=1, stride=1, act=None)
        )

        self.upconv3_res = nn.Sequential(
            conv(ch4 * 2, ch4, kernel_size=ks, stride=1),
            conv(ch4, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)
        )

        self.img_d4_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch3, kernel_size=ks, stride=1),
        )

        self.upconv2_kernel = nn.Sequential(
            conv(ch3 * 2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch3, kernel_size=ks, stride=1),
            conv(ch3, self.kernel_dim, kernel_size=1, stride=1, act=None)
        )

        self.upconv2_res = nn.Sequential(
            conv(ch3 * 2, ch3, kernel_size=ks, stride=1),
            conv(ch3, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)
        )
        self.img_d2_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1)
        )

        self.img_d1_feature = nn.Sequential(
            conv(3, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch1, kernel_size=ks, stride=1),
        )

        self.upconv1_kernel = nn.Sequential(
            conv(ch2 * 2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, self.kernel_dim, kernel_size=1, stride=1, act=None)
        )

        self.upconv1_res = nn.Sequential(
            conv(ch2 * 2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)
        )

        self.upconv0_kernel = nn.Sequential(
            conv(ch1 * 2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, self.kernel_dim, kernel_size=1, stride=1, act=None)
        )

        self.upconv0_res = nn.Sequential(
            conv(ch1 * 2, ch2, kernel_size=ks, stride=1),
            conv(ch2, ch2, kernel_size=ks, stride=1),
            conv(ch2, 3, kernel_size=1, stride=1)
        )

        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path), strict=False)

    ##########################################################################
    def forward(self, C, fs):
        img_d8 = F.interpolate(C[:, :3], scale_factor=1 / 8, mode='area')
        img_d8_feature = self.img_d8_feature(img_d8)
        feature_d8 = torch.cat([fs[0], img_d8_feature], 1)  # ch4*2
        kernel_d8 = self.upconv3_kernel(feature_d8)

        res_f8 = self.upconv3_res(feature_d8)

        est_img_d8 = img_d8 + FAC(img_d8, kernel_d8, self.kernel_width) + res_f8

        est_img_d4_interpolate = F.interpolate(est_img_d8, scale_factor=2, mode='area')

        img_d4_feature = self.img_d4_feature(est_img_d4_interpolate)
        feature_d4 = torch.cat([fs[1], img_d4_feature], 1)
        kernel_d4 = self.upconv2_kernel(feature_d4)

        res_f4 = self.upconv2_res(feature_d4)

        est_img_d4 = est_img_d4_interpolate + FAC(est_img_d4_interpolate, kernel_d4, self.kernel_width) + res_f4

        est_img_d2_interpolate = F.interpolate(est_img_d4, scale_factor=2, mode='area')

        img_d2_feature = self.img_d2_feature(est_img_d2_interpolate)
        feature_d2 = torch.cat([fs[2], img_d2_feature], 1)

        kernel_d2 = self.upconv1_kernel(feature_d2)
        res_f2 = self.upconv1_res(feature_d2)

        est_img_d2 = est_img_d2_interpolate + FAC(est_img_d2_interpolate, kernel_d2, self.kernel_width) + res_f2

        est_img_d1_interploate = F.interpolate(est_img_d2, scale_factor=2, mode='area')

        img_d1_feature = self.img_d1_feature(est_img_d1_interploate)
        feature_d1 = torch.cat([fs[3], img_d1_feature], 1)
        kernel_d1 = self.upconv0_kernel(feature_d1)

        res_f1 = self.upconv0_res(feature_d1)

        est_img_d1 = est_img_d1_interploate + FAC(est_img_d1_interploate, kernel_d1, self.kernel_width) + res_f1

        est_img_d1_ = torch.clip(est_img_d1, -1.0, 1.0)

        return est_img_d1_


class SBTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature14to160 = DRBFeature()
        self.feature18to160 = DRBFeature()
        self.feature14to18 = DRBFeature()
        self.feature18to18 = DRBFeature()
        self.feature18to14 = DRBFeature()
        self.feature160to18 = DRBFeature()
        self.feature160to14 = DRBFeature()
        self.kernel = DRBKernel()
        self.alphanet = AlphaNet()
        self.refinenet = RefineNet()

    def forward(self, src, src_lens_type, tgt_lens_type, src_F, tgt_F, disparity, cateye_coord, use_alpha):
        x = src * 2 - 1
        x_ori = x
        h, w = x.shape[2:]

        x = F.interpolate(x, size=(h//2, w//2), mode='bilinear', align_corners=True)
        cateye_coord = F.interpolate(cateye_coord, size=(h//2, w//2), mode='bilinear', align_corners=True)

        if use_alpha:
            alpha = self.alphanet(x)
        else:
            alpha = torch.zeros_like(x[:, :1])

        x = torch.cat([
            x,
            alpha,
            cateye_coord,
            torch.ones_like(x[:, :1]) * src_lens_type[:, None, None, None],
            torch.ones_like(x[:, :1]) * tgt_lens_type[:, None, None, None],
        ], dim=1)

        if src_F[0] == 1.4 and tgt_F[0] == 16.0:
            x = self.kernel(x, self.feature14to160(x))
        elif src_F[0] == 1.8 and tgt_F[0] == 16.0:
            x = self.kernel(x, self.feature18to160(x))
        elif src_F[0] == 1.4 and tgt_F[0] == 1.8:
            x = self.kernel(x, self.feature14to18(x))
        elif src_F[0] == 1.8 and tgt_F[0] == 1.8:
            x = self.kernel(x, self.feature18to18(x))
        elif src_F[0] == 1.8 and tgt_F[0] == 1.4:
            x = self.kernel(x, self.feature18to14(x))
        elif src_F[0] == 16.0 and tgt_F[0] == 1.8:
            x = self.kernel(x, self.feature160to18(x))
        elif src_F[0] == 16.0 and tgt_F[0] == 1.4:
            x = self.kernel(x, self.feature160to14(x))
        else:
            print('Use interpolation.')
            blur_ratio = src_F / tgt_F
            if 14 / 160 <= blur_ratio < 18 / 160:
                fs1 = self.feature14to160(x)
                fs2 = self.feature18to160(x)
                ratio = (blur_ratio - 14 / 160) / (18 / 160 - 14 / 160)
            elif 18 / 160 <= blur_ratio < 14 / 18:
                fs1 = self.feature18to160(x)
                fs2 = self.feature14to18(x)
                ratio = (blur_ratio - 18 / 160) / (14 / 18 - 18 / 160)
            elif 14 / 18 <= blur_ratio < 18 / 18:
                fs1 = self.feature14to18(x)
                fs2 = self.feature18to18(x)
                ratio = (blur_ratio - 14 / 18) / (18 / 18 - 14 / 18)
            elif 18 / 18 <= blur_ratio < 18 / 14:
                fs1 = self.feature18to18(x)
                fs2 = self.feature18to14(x)
                ratio = (blur_ratio - 18 / 18) / (18 / 14 - 18 / 18)
            elif 18 / 14 <= blur_ratio < 160 / 18:
                fs1 = self.feature18to14(x)
                fs2 = self.feature160to18(x)
                ratio = (blur_ratio - 18 / 14) / (160 / 18 - 18 / 14)
            elif 160 / 18 <= blur_ratio <= 160 / 14:
                fs1 = self.feature160to18(x)
                fs2 = self.feature160to14(x)
                ratio = (blur_ratio - 160 / 18) / (160 / 14 - 160 / 18)
            else:
                raise Exception('The ratio of f-number of source image and target image is out of range.')

            feature = [(1 - ratio) * fs1[0] + ratio * fs2[0],
                       (1 - ratio) * fs1[1] + ratio * fs2[1],
                       (1 - ratio) * fs1[2] + ratio * fs2[2],
                       (1 - ratio) * fs1[3] + ratio * fs2[3]]
            x = self.kernel(x, feature)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        alpha = F.interpolate(alpha, size=(h, w), mode='bilinear', align_corners=True)
        x = x + self.refinenet(x, alpha, x_ori)

        x = 0.5 * x + 0.5

        return x, alpha

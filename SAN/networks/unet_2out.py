""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.mynet_parts.normlizations import get_norm_layer
from networks.basic_unet_3d import UnetEncoder, Up


class UNet_2out(nn.Module):
    def __init__(self, n_in=1, n_out=1, first_channels=64, n_dps=4, use_bilinear=True, use_pool=True,
                 norm_type='instance3D', **kwargs):
        super(UNet_2out, self).__init__()

        norm_layer = get_norm_layer(norm_type)

        self.encoder = UnetEncoder(n_in, first_channels, n_dps, use_pool, norm_layer)
        first_channels = first_channels * pow(2, n_dps)
        self.decoder = UnetDecoder_3out(n_out, first_channels, n_dps, use_bilinear, norm_layer)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out


class UnetDecoder_3out(nn.Module):
    def __init__(self, n_classes, first_channels, n_dps, use_bilinear, norm_layer, is_out_features=False):
        super(UnetDecoder_3out, self).__init__()

        self.up_blocks = nn.ModuleList()
        T_channels = first_channels
        out_channels = T_channels // 2
        in_channels = T_channels + out_channels

        for i in range(n_dps):
            self.up_blocks.append(Up(T_channels, in_channels, out_channels, use_bilinear, norm_layer))
            T_channels = out_channels
            out_channels = T_channels // 2
            in_channels = T_channels + out_channels
        # one more divide in out_channels
        self.outc1 = nn.Conv3d(out_channels*2, n_classes, kernel_size=1)
        self.outc2 = nn.Conv3d(out_channels*2, n_classes, kernel_size=1)

        self.is_out_features = False

    def forward(self, features):
        pos_feat = len(features) - 1
        x = features[pos_feat]
        out_features = [x]
        for up_block in self.up_blocks:
            pos_feat -= 1
            x = up_block(x, features[pos_feat])
            out_features.append(x)
        x1 = self.outc1(x)
        x2 = self.outc2(x)
        if self.is_out_features:
            return x1,x2, out_features
        else:
            return x1,x2


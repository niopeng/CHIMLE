import math
import os
from collections import namedtuple
import torch
import torch.nn as nn
from torchvision import models as tv
from . import block as B


####################
# Generator
####################
class TIM(nn.Module):
    def __init__(self, in_nc, code_nc, out_nc, num_residual_channels, num_dense_channels, num_blocks, num_levels=4,
                 act_type='leakyrelu', upsample_kernel_mode="nearest", feat_scales=None, map_nc=128, latent_nc=512,
                 task='Colourization'):
        super(TIM, self).__init__()
        self.num_levels = num_levels
        self.code_nc = code_nc
        self.feat_scales = feat_scales if feat_scales is not None else [0.1] * (self.num_levels - 1)
        self.out_layer = B.RerangeLayer()
        self.map_nc = map_nc
        self.task = task

        for i in range(self.num_levels):
            cur_num_dc = num_dense_channels[i]
            cur_num_rc = num_residual_channels[i]

            # mapping network
            mapping_net = [B.sequential(nn.Linear(map_nc, latent_nc), B.act(act_type))]
            self.add_module("level_%d_map_0" % (i + 1), B.sequential(*mapping_net))
            mapping_net = []
            for _ in range(7):
                if i > 0 and _ == 0:
                    mapping_net.append(B.sequential(nn.Linear(latent_nc * (i+1), latent_nc), B.act(act_type)))
                else:
                    mapping_net.append(B.sequential(nn.Linear(latent_nc, latent_nc), B.act(act_type)))
            self.add_module("level_%d_map" % (i + 1), B.sequential(*mapping_net))

            if i == 0:
                fea_conv = B.conv_block(in_nc + code_nc, cur_num_rc, kernel_size=3, act_type=None)
            else:
                fea_conv = B.conv_block(in_nc + code_nc + num_residual_channels[i - 1], cur_num_rc, kernel_size=3,
                                        act_type=None)

            # RRDB blocks
            rb_blocks = [B.RRDB(cur_num_rc, kernel_size=3, gc=cur_num_dc, bias=True, pad_type='zero',
                                act_type=act_type) for _ in range(num_blocks)]
            transformations = [nn.Linear(latent_nc, 2 * cur_num_rc) for _ in range(num_blocks)]
            lr_conv = B.conv_block(cur_num_rc, cur_num_rc, kernel_size=3, act_type=None)

            style_block = B.StyleBlock(rb_blocks, transformations, lr_conv)
            # The layer that produces the feature to concatenate with the next level
            hr_conv = B.conv_block(cur_num_rc, cur_num_rc, kernel_size=3, act_type=act_type)
            out_conv = B.conv_block(cur_num_rc, out_nc, kernel_size=3, act_type="tanh")

            if i == self.num_levels - 1 and task == "Colourization":
                layer = B.conv_block(cur_num_rc, cur_num_rc, kernel_size=3, act_type=act_type)
            else:
                layer = B.upconv_block(cur_num_rc, cur_num_rc, act_type=act_type, mode=upsample_kernel_mode)

            self.add_module("level_%d_feat" % (i + 1), fea_conv)
            self.add_module("level_%d_style" % (i + 1), B.ShortcutBlock(style_block))
            self.add_module("level_%d_up" % (i + 1), B.sequential(layer, hr_conv))
            self.add_module("level_%d_out" % (i + 1), out_conv)

    def forward(self, net_input, codes):
        if self.task == "Colourization":
            return self.forward_colourization(net_input=net_input, codes=codes)
        elif self.task == "Super_Resolution":
            return self.forward_super_resolution(net_input=net_input, codes=codes)
        else:
            return self.forward_generic(net_input=net_input, codes=codes)

    def forward_colourization(self, net_input, codes):
        assert len(codes) <= self.num_levels, "Number of codes should be no more than number of level of the network"
        outputs = []
        feature = None
        map_features = []
        for i, code in enumerate(codes):
            bs, _, w, h = net_input[i].shape
            if i == 0:
                x = torch.cat((net_input[i], code[:, self.map_nc:].reshape(bs, self.code_nc, w, h)), dim=1)
            else:
                # concat with the previous level output and feature
                x = torch.cat((net_input[i], code[:, self.map_nc:].reshape(bs, self.code_nc, w, h),
                               feature * self.feat_scales[i - 1]), dim=1)

            mapped_code = getattr(self, "level_%d_map_0" % (i + 1))(code[:, :self.map_nc])
            if i > 0:
                map_feature = getattr(self, "level_%d_map" % (i + 1))(torch.cat([mapped_code] + map_features, dim=1))
            else:
                map_feature = getattr(self, "level_%d_map" % (i + 1))(mapped_code)
            map_features.append(map_feature)
            feature = getattr(self, "level_%d_feat" % (i + 1))(x)
            feature = getattr(self, "level_%d_style" % (i + 1))(feature, map_feature)
            if i == self.num_levels - 1:
                feature = getattr(self, "level_%d_up" % (i + 1))(feature)
            out = getattr(self, "level_%d_out" % (i + 1))(feature)

            if i < self.num_levels - 1:
                feature = getattr(self, "level_%d_up" % (i + 1))(feature)

            outputs.append(out)

        for i in range(0, len(outputs)):
            outputs[i] = torch.cat((net_input[i], outputs[i]), 1)
        return outputs

    def forward_super_resolution(self, net_input, codes):
        assert len(codes) <= self.num_levels, "Number of codes should be no more than number of level of the network"
        outputs = []
        feature = None
        out = None
        map_features = []
        for i, code in enumerate(codes):
            if i == 0:
                bs, _, w, h = net_input[0].shape
                x = torch.cat((net_input[0], code[:, self.map_nc:].reshape(bs, self.code_nc, w, h)), dim=1)
            else:
                bs, _, w, h = out.shape
                # concat with the previous level output and feature
                x = torch.cat((out, code[:, self.map_nc:].reshape(bs, self.code_nc, w, h), feature *
                               self.feat_scales[i - 1]), dim=1)
            mapped_code = getattr(self, "level_%d_map_0" % (i + 1))(code[:, :self.map_nc])
            if i > 0:
                map_feature = getattr(self, "level_%d_map" % (i + 1))(torch.cat([mapped_code] + map_features, dim=1))
            else:
                map_feature = getattr(self, "level_%d_map" % (i + 1))(mapped_code)
            map_features.append(map_feature)
            feature = getattr(self, "level_%d_feat" % (i + 1))(x)
            feature = getattr(self, "level_%d_style" % (i + 1))(feature, map_feature)
            feature = getattr(self, "level_%d_up" % (i + 1))(feature)
            out = getattr(self, "level_%d_out" % (i + 1))(feature)
            outputs.append(self.out_layer(out))
        return outputs

    def forward_generic(self, net_input, codes):
        assert len(codes) <= self.num_levels, "Number of codes should be no more than number of level of the network"
        outputs = []
        feature = None
        map_features = []
        for i, code in enumerate(codes):
            bs, _, w, h = net_input[i].shape
            if i == 0:
                x = torch.cat((net_input[i], code[:, self.map_nc:].reshape(bs, self.code_nc, w, h)), dim=1)
            else:
                x = torch.cat((net_input[i], code[:, self.map_nc:].reshape(bs, self.code_nc, w, h),
                               feature * self.feat_scales[i - 1]), dim=1)
            mapped_code = getattr(self, "level_%d_map_0" % (i + 1))(code[:, :self.map_nc])
            if i > 0:
                map_feature = getattr(self, "level_%d_map" % (i + 1))(torch.cat([mapped_code] + map_features, dim=1))
            else:
                map_feature = getattr(self, "level_%d_map" % (i + 1))(mapped_code)
            map_features.append(map_feature)
            feature = getattr(self, "level_%d_feat" % (i + 1))(x)
            feature = getattr(self, "level_%d_style" % (i + 1))(feature, map_feature)
            out = getattr(self, "level_%d_out" % (i + 1))(feature)
            outputs.append(self.out_layer(out))
            if i < self.num_levels - 1:
                feature = getattr(self, "level_%d_up" % (i + 1))(feature)
        return outputs


# Learned perceptual network, modified from https://github.com/richzhang/PerceptualSimilarity
class LPNet(nn.Module):
    def __init__(self, pnet_type='vgg', version='0.1'):
        super(LPNet, self).__init__()

        self.scaling_layer = B.ScalingLayer()
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.L = 5
        self.lins = [B.NetLinLayer() for _ in range(self.L)]

        model_path = os.path.abspath(
            os.path.join('.', 'models/weights/v%s/%s.pth' % (version, pnet_type)))
        print('Loading model from: %s' % model_path)
        weights = torch.load(model_path)
        for i in range(self.L):
            self.lins[i].weight = torch.sqrt(weights["lin%d.model.1.weight" % i])

    def forward(self, in0, avg=False):
        in0 = 2 * in0 - 1
        in0_input = self.scaling_layer(in0)
        outs0 = self.net.forward(in0_input)
        feats0 = {}
        shapes = []
        res = []

        for kk in range(self.L):
            feats0[kk] = B.normalize_tensor(outs0[kk])

        if avg:
            res = [self.lins[kk](feats0[kk]).mean([2, 3], keepdim=False) for kk in range(self.L)]
        else:
            for kk in range(self.L):
                cur_res = self.lins[kk](feats0[kk])
                shapes.append(cur_res.shape[-1])
                res.append(cur_res.reshape(cur_res.shape[0], -1))

        return res, shapes


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out

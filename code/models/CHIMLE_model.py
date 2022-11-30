from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import models.networks as networks
from .base_model import BaseModel
import math
import os
from utils import util


def compute_feature_loss(gen_feat, real_feat, gen_shape):
    # compute l2 feature loss given features
    result = 0
    for i, g_feat in enumerate(gen_feat):
        cur_diff = torch.sum((g_feat - real_feat[i]) ** 2, dim=1) / (gen_shape[i] ** 2)
        result += cur_diff
    return result


class CHIMLEModel(BaseModel):
    def __init__(self, opt):
        super(CHIMLEModel, self).__init__(opt)
        train_opt = opt['train']
        self.task = opt['task']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt)
        if self.is_train:
            self.netG.train()
        self.load()
        # store the number of levels and code channel
        self.num_levels = opt['levels']
        self.code_nc = opt['network_G']['code_nc']
        self.map_nc = opt['network_G']['map_nc']

        # define losses, optimizer and scheduler
        # G pixel loss
        if train_opt is not None and train_opt['pixel_weight'] > 0:
            l_pix_type = train_opt['pixel_criterion']
            if l_pix_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif l_pix_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
            self.l_pix_w = train_opt['pixel_weight']
        else:
            print('Remove pixel loss.')
            self.cri_pix = None

        self.netF = networks.define_F().cuda()
        self.projections = None
        if self.is_train:
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            map_network_params = []
            core_network_params = []
            # can freeze weights for any of the levels
            freeze_level = train_opt['freeze_level']
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    if freeze_level:
                        if "level_%d" % freeze_level not in k:
                            if 'map' in k:
                                map_network_params.append(v)
                            else:
                                core_network_params.append(v)
                    else:
                        if 'map' in k:
                            map_network_params.append(v)
                        else:
                            core_network_params.append(v)
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))
            map_multiplier = 1e-2 if "map_multiplier" not in train_opt else train_opt['map_multiplier']
            print("Mapping network param multiplier: %f with total length %d" % (map_multiplier, len(map_network_params)))
            self.optimizer_G = torch.optim.Adam([{'params': core_network_params},
                                                 {'params': map_network_params, 'lr': map_multiplier * train_opt['lr_G']}],
                                                lr=train_opt['lr_G'], weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)
            # for resume training - load the previous optimizer stats
            load_checkpoint_netG = True if self.opt['path']['pretrain_model_G'] is None else False
            self.load_optimizer(load_netG=load_checkpoint_netG)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, train_opt['lr_steps'],
                                                                    train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def feed_data(self, data, code=[], need_HR=True):
        self.network_input = []
        for lr_res in data['network_input']:
            self.network_input.append(lr_res.to(self.device))
        self.code = code
        if need_HR:  # train or val
            self.targets = dict()
            # only feed the images, not their paths
            for key, val in data.items():
                if ('HR' in key or 'D' in key) and 'path' not in key:
                    self.targets[key] = val.to(self.device)  # original colored image in RGB color space

    # Generate random code input at specified level (if left empty, then generate code for all levels)
    def gen_code(self, bs, w, h, levels=[], tensor_type=torch.randn):
        gen_levels = levels if levels != [] else range(self.num_levels)
        out_code = []
        for i in gen_levels:
            out_code.append(tensor_type(bs, self.map_nc + self.code_nc * w * (2 ** i) * h * (2 ** i),
                                        device=self.device))
        return out_code

    # intermediate supervision adds loss from intermediate resolutions
    def _get_feature_loss(self, gen_img_rgb, real_img_rgb):
        gen_feat, gen_shape = self.netF(gen_img_rgb)
        real_feat, real_shape = self.netF(real_img_rgb)
        return compute_feature_loss(gen_feat, real_feat, gen_shape)

    # Random projection matrix for reducing LPIPS feature dimension
    def init_projection(self, h, w, total_dim=1000):
        self.projections = None
        fake_input = torch.zeros(1, 3, h, w).to(self.device)
        fake_feat, fake_shape = self.netF(fake_input)
        self.projections = []
        dim_per_layer = int(total_dim * 1. / len(fake_feat))
        for feat in fake_feat:
            self.projections.append(F.normalize(torch.randn(feat.shape[1], dim_per_layer, device=self.device), p=2, dim=1))

    def clear_projection(self):
        self.projections = None

    def _get_target_at_level(self, level):
        for key, val in self.targets.items():
            if str(level + 1) in key and 'path' not in key:
                return val
        return self.targets['HR']

    def get_features(self, level=-1):
        '''
        Assuming the generated features are for the same LR input, therefore just one pass for the target feature
        '''
        self.netG.eval()
        out_dict = OrderedDict()
        with torch.no_grad():
            gen_imgs = self.netG(self.network_input, self.code)
            net_f_inp = gen_imgs[level]
            if self.task == 'Colourization':
                net_f_inp = util.lab2rgb_tensor(net_f_inp)
            gen_feat, gen_shape = self.netF(net_f_inp)
            real_feat, real_shape = self.netF(self._get_target_at_level(level))

            gen_features = []
            real_features = []
            # random projection
            for i, g_feat in enumerate(gen_feat):
                proj_gen_feat = torch.mm(g_feat, self.projections[i])
                proj_real_feat = torch.mm(real_feat[i], self.projections[i])
                gen_features.append(proj_gen_feat / gen_shape[i])
                real_features.append(proj_real_feat / gen_shape[i])
            gen_features = torch.cat(gen_features, dim=1)
            real_features = torch.cat(real_features, dim=1)
            out_dict['gen_feat'] = gen_features.to(self.device)
            out_dict['real_feat'] = real_features.to(self.device)

        self.netG.train()
        return out_dict

    def get_loss(self, level=-1):
        self.netG.eval()
        with torch.no_grad():
            gen_imgs = self.netG(self.network_input, self.code)
            img_out_rgb = gen_imgs[level]
            if self.task == 'Colourization':
                img_out_rgb = util.lab2rgb_tensor(img_out_rgb)

            real_img = self._get_target_at_level(level)  # ground-truth image

            if self.cri_pix:  # pixel loss
                result = self.l_pix_w * self.cri_pix(img_out_rgb, real_img)
            else:
                result = self._get_feature_loss(img_out_rgb, real_img)
        self.netG.train()
        return result

    def optimize_parameters(self, inter_supervision=True):
        torch.autograd.set_detect_anomaly(True)
        self.optimizer_G.zero_grad()
        outputs = self.netG(self.network_input, self.code)
        l_g_total = 0.

        if inter_supervision:
            for i, output in enumerate(outputs):
                img_out_rgb = output
                if self.task == 'Colourization':
                    img_out_rgb = util.lab2rgb_tensor(img_out_rgb)
                real_img = self._get_target_at_level(i)  # ground-truth image

                if self.cri_pix:  # pixel loss
                    l_g_total += self.l_pix_w * self.cri_pix(img_out_rgb.to(self.device), real_img.to(self.device))
                else:
                    l_g_total += self._get_feature_loss(img_out_rgb.to(self.device), real_img.to(self.device))
        else:
            img_out_rgb = outputs[-1]
            if self.task == 'Colourization':
                img_out_rgb = util.lab2rgb_tensor(img_out_rgb)
            real_img = self._get_target_at_level(-1)

            if self.cri_pix:  # pixel loss
                l_g_total += self.l_pix_w * self.cri_pix(img_out_rgb.to(self.device), real_img.to(self.device))
            else:
                l_g_total += self._get_feature_loss(img_out_rgb.to(self.device), real_img.to(self.device))

        l_g_total = torch.sum(l_g_total)
        l_g_total.backward()
        self.optimizer_G.step()
        if self.opt['train']['pixel_weight'] > 0:
            self.log_dict[self.opt['train']['pixel_criterion']] = l_g_total.item()
        else:
            self.log_dict['l_g_lpips'] = l_g_total.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            output = self.netG(self.network_input, self.code)
            # predicted image is in rgb format
            self.pred = []
            img_out = output[-1]
            if self.task == 'Colourization':
                img_out = util.lab2rgb_tensor(img_out)
            self.pred.append(img_out)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['HR_pred'] = self.pred[-1].detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.targets['HR'].detach()[0].float().cpu()  # original image in rgb
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            message = '-------------- Generator --------------\n' + s + '\n'
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            with open(network_path, 'w') as f:
                f.write(message)

            # F, Perceptual Network
            s, n = self.get_network_description(self.netF)
            print('Number of parameters in F: {:,d}'.format(n))
            message = '\n\n\n-------------- Perceptual Network --------------\n' + s + '\n'
            with open(network_path, 'a') as f:
                f.write(message)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            print('loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

    def load_optimizer(self, load_netG=True):
        load_path_O = os.path.join(self.save_dir, 'latest.tar') if self.opt['path']['pretrain_model_O'] is None else \
            self.opt['path']['pretrain_model_O']
        if load_path_O is not None and os.path.isfile(load_path_O):
            print('Loading training checkpoint from [{:s}] ...'.format(load_path_O))
            checkpoint = torch.load(load_path_O, map_location=self.device)
            self.data_idx = checkpoint['data_idx']
            self.iteration = checkpoint['iteration']
            if load_netG:
                self.netG.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])

    def save(self, iter_label, data_idx=0):
        net_G_stat = self.save_network(self.save_dir, self.netG, 'G', iter_label)
        torch.save({
            'data_idx': data_idx,
            'iteration': int(iter_label),
            'model_state_dict': net_G_stat,
            'optimizer_state_dict': self.optimizer_G.state_dict(),
        }, os.path.join(self.save_dir, 'latest.tar'))

import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import data.util as util
from utils import util as img_util


class LRHRDataset(data.Dataset):
    '''
    Read input and target images.
    Generates intermediate input or target images on-the-fly.
    The group is matched by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_HR = None
        self.LR_env = None  # environment for lmdb
        self.HR_env = None

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_HR = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n')) for line in f])
            if opt['dataroot_LR'] is not None:
                raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
        else:  # read image list from lmdb or image files
            self.HR_env, self.paths_HR = util.get_image_paths(opt['data_type'], opt['dataroot_HR'])
            self.LR_env, self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])

        assert self.paths_HR, 'Error: HR path is empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                'HR and LR datasets have different number of images - {}, {}.'.format(len(self.paths_LR),
                                                                                      len(self.paths_HR))

    def __getitem__(self, index):
        HR_path = self.paths_HR[index]
        LR_path = self.paths_LR[index]

        img_HR = util.read_img(self.HR_env, HR_path)
        img_LR_3 = util.read_img(self.LR_env, LR_path)
        scale = img_HR.shape[0] // img_LR_3.shape[0]

        H, W, C = img_HR.shape
        center_crop = False if self.opt["center_crop"] not in self.opt else self.opt["center_crop"]
        crop_size = self.opt['crop_size'] if "crop_size" in self.opt else img_HR.shape[0]

        if center_crop:
            rnd_h = (H - crop_size) // 2
            rnd_w = (W - crop_size) // 2
        else:
            # randomly crop
            rnd_h = random.randint(0, max(0, H - crop_size))
            rnd_w = random.randint(0, max(0, W - crop_size))

        img_HR = img_HR[rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :]

        # If LR and HR have identical resolution, scale input resolution down. Used in generic case
        if scale == 1:
            # crop the input image first
            img_LR_3 = img_LR_3[rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :]
            # augmentation - flip, rotate
            img_LR_3, img_HR = util.augment([img_LR_3, img_HR], self.opt['use_flip'], self.opt['use_rot'])
            # generate downsampled inputs
            img_LR_2 = img_util.downsample_PIL(img_LR_3, scale=1. / 2)
            img_LR_1 = img_util.downsample_PIL(img_LR_3, scale=1. / 4)
            img_LR = img_util.downsample_PIL(img_LR_3, scale=1. / 8)
        else:  # used in super resolution
            # low resolution input cropping
            rnd_h = rnd_h // scale
            rnd_w = rnd_w // scale
            crop_size = crop_size // scale
            img_LR_3 = img_LR_3[rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :]
            # augmentation - flip, rotate
            img_LR_3, img_HR = util.augment([img_LR_3, img_HR], self.opt['use_flip'], self.opt['use_rot'])
            img_LR_1 = img_LR_2 = img_LR = img_LR_3
        img_D3 = img_util.downsample_PIL(img_HR, scale=1. / 2)
        img_D2 = img_util.downsample_PIL(img_HR, scale=1. / 4)
        img_D1 = img_util.downsample_PIL(img_HR, scale=1. / 8)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
            img_LR_1 = img_LR_1[:, :, [2, 1, 0]]
            img_LR_2 = img_LR_2[:, :, [2, 1, 0]]
            img_LR_3 = img_LR_3[:, :, [2, 1, 0]]
            img_D1 = img_D1[:, :, [2, 1, 0]]
            img_D2 = img_D2[:, :, [2, 1, 0]]
            img_D3 = img_D3[:, :, [2, 1, 0]]
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()
        img_LR_1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR_1, (2, 0, 1)))).float()
        img_LR_2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR_2, (2, 0, 1)))).float()
        img_LR_3 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR_3, (2, 0, 1)))).float()
        img_D1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_D1, (2, 0, 1)))).float()
        img_D2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_D2, (2, 0, 1)))).float()
        img_D3 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_D3, (2, 0, 1)))).float()

        return {'network_input': [img_LR, img_LR_1, img_LR_2, img_LR_3],
                'HR': img_HR,
                'LR_path': LR_path,
                'HR_path': HR_path,
                'D1': img_D1,
                'D2': img_D2,
                'D3': img_D3,
                'is_valid': True}

    def __len__(self):
        return len(self.paths_HR)

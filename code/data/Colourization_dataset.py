import os.path
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
from utils import util as img_util


class ColourizationDataset(data.Dataset):
    '''
    Given target coloured images, generates gray-scale input and intermediate resolution images on-the-fly.
    The group is matched by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(ColourizationDataset, self).__init__()
        self.opt = opt
        self.paths_HR_Color = None
        self.HR_Color_env = None

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_HR_Color = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n')) for line in f])
        else:  # read image list from lmdb or image files
            self.HR_Color_env, self.paths_HR_Color = util.get_image_paths(opt['data_type'], opt['dataroot_HR'])

        assert self.paths_HR_Color, 'Error: Target image folder path is empty.'

    def __getitem__(self, index):

        validity = True
        # generating outputs for the specified IDs in the test set rather than generating outputs for the whole test set
        if self.opt['phase'] != 'train':
            if 'target_images_id' in self.opt:
                if self.paths_HR_Color[index] not in self.opt['target_images_id']:
                    return {'is_valid': False}

        # get HR_Color image
        HR_path = self.paths_HR_Color[index]
        img_HR_bgr = util.read_img(self.HR_Color_env, HR_path)  # HWC, BGR, [0,1], [0, 255]

        # force to 3 channels
        if img_HR_bgr.ndim == 2:
            img_HR_bgr = cv2.cvtColor(img_HR_bgr, cv2.COLOR_GRAY2BGR)

        img_HR_rgb = cv2.cvtColor(img_HR_bgr, cv2.COLOR_BGR2RGB)  # HWC, RGB, [0, 1], 256 * 256

        # augmentation - flip, rotate
        if self.opt['phase'] == 'train':
            [img_HR_rgb] = util.augment([img_HR_rgb], self.opt['use_flip'], self.opt['use_rot'])

        # down-sampling on-the-fly
        # HWC, RGB, [0, 1]

        img_D1_rgb = img_util.downsample_PIL(img_HR_rgb, scale=1. / 8)  # 32  * 32
        img_D2_rgb = img_util.downsample_PIL(img_HR_rgb, scale=1. / 4)  # 64  * 64
        img_D3_rgb = img_util.downsample_PIL(img_HR_rgb, scale=1. / 2)  # 128 * 128

        # L channel
        img_HR_lab = img_util.rgb2lab(img_HR_rgb)
        img_D1_lab = img_util.rgb2lab(img_D1_rgb)
        img_D2_lab = img_util.rgb2lab(img_D2_rgb)
        img_D3_lab = img_util.rgb2lab(img_D3_rgb)

        HR_L_channel = img_HR_lab[:, :, 0] / 100.0
        D1_L_channel = img_D1_lab[:, :, 0] / 100.0
        D2_L_channel = img_D2_lab[:, :, 0] / 100.0
        D3_L_channel = img_D3_lab[:, :, 0] / 100.0

        HR_L_channel_tensor = torch.Tensor(HR_L_channel)[None, :, :]
        D1_L_channel_tensor = torch.Tensor(D1_L_channel)[None, :, :]
        D2_L_channel_tensor = torch.Tensor(D2_L_channel)[None, :, :]
        D3_L_channel_tensor = torch.Tensor(D3_L_channel)[None, :, :]

        # HWC to CHW, numpy to tensor
        img_HR_tensor_rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR_rgb, (2, 0, 1)))).float()
        img_D1_tensor_rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_D1_rgb, (2, 0, 1)))).float()
        img_D2_tensor_rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_D2_rgb, (2, 0, 1)))).float()
        img_D3_tensor_rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(img_D3_rgb, (2, 0, 1)))).float()

        if self.opt['phase'] == 'train':
            res = {
                'network_input': [D1_L_channel_tensor, D2_L_channel_tensor, D3_L_channel_tensor, HR_L_channel_tensor],
                'HR': img_HR_tensor_rgb,
                'LR_path': HR_path,
                'HR_path': HR_path,
                'D1': img_D1_tensor_rgb,
                'D2': img_D2_tensor_rgb,
                'D3': img_D3_tensor_rgb,
                'D4': img_HR_tensor_rgb}
        else:
            res = {
                'network_input': [D1_L_channel_tensor, D2_L_channel_tensor, D3_L_channel_tensor, HR_L_channel_tensor],
                'HR': img_HR_tensor_rgb,
                'LR_path': HR_path,
                'HR_path': HR_path,
                'is_valid': validity}
        return res

    def __len__(self):
        return len(self.paths_HR_Color)

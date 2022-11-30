import numpy as np
import torch
import torch.utils.data as data
import data.util as util


class LRDataset(data.Dataset):
    '''Read LR images only in the test phase.'''

    def __init__(self, opt):
        super(LRDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.LR_env = None  # environment for lmdb

        # read image list from lmdb or image files
        self.LR_env, self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])
        assert self.paths_LR, 'Error: LR paths are empty.'

    def __getitem__(self, index):
        # get LR image
        LR_path = self.paths_LR[index]
        img_LR = util.read_img(self.LR_env, LR_path)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LR.shape[2] == 3:
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        return {'network_input': [img_LR],
                'LR_path': LR_path,
                'HR_path': LR_path,
                'is_valid': True}

    def __len__(self):
        return len(self.paths_LR)

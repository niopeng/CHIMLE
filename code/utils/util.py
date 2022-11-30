import os
import math
from datetime import datetime
import numpy as np
import cv2
import torch
from skimage import color
from torchvision.utils import make_grid
import PIL
from PIL import Image

####################
# color space config
####################
color_output_mode = "A*L*B*L"  # "A*L*B*L" or "AB"
AB_range = "standard"  # "standard" or "real"

l_center = 0.0
l_norm = 100.0

if AB_range == "standard":
    a_center = 0.0
    a_norm = 127.0
    b_center = 0.0
    b_norm = 127.0
elif AB_range == "real":
    a_center = 6.0345
    a_norm = 92.2195
    b_center = - 6.6905
    b_norm = 101.1725


####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


####################
# image convert
####################


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def numpy2tensor(inp):
    """
    Parameters:
        inp : is the numpy array, R G B channels order
    Returns:
        3 dimension tensor, shape = (C, H, W)
    """
    # BGR to RGB, HWC to CHW, numpy to tensor
    if inp.shape[2] == 3:
        inp = inp[:, :, [2, 1, 0]]
    return torch.from_numpy(np.ascontiguousarray(np.transpose(inp, (2, 0, 1)))).float()


def rgb2lab(rgb_image):
    """
    Parameters:
        rgb_image : 3d numpy array, H x W x C
    Returns:
        lab_image: 3d numpy array, H x W x C
    Explanation:
        Convert the RGB input image to the LAB image using the color library
    """
    return color.rgb2lab(rgb_image)


def rgb2xyz(rgb):
    mask = (rgb > .04045).type(torch.FloatTensor)
    if rgb.is_cuda:
        mask = mask.cuda()

    rgb = (((rgb + .055) / 1.055) ** 2.4) * mask + rgb / 12.92 * (1 - mask)

    x = .412453 * rgb[:, 0, :, :] + .357580 * rgb[:, 1, :, :] + .180423 * rgb[:, 2, :, :]
    y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] + .072169 * rgb[:, 2, :, :]
    z = .019334 * rgb[:, 0, :, :] + .119193 * rgb[:, 1, :, :] + .950227 * rgb[:, 2, :, :]
    out = torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)

    return out


def xyz2rgb(xyz):
    r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] - 0.49853633 * xyz[:, 2, :, :]
    g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] + .04155593 * xyz[:, 2, :, :]
    b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] + 1.05731107 * xyz[:, 2, :, :]

    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
    rgb = torch.max(rgb, torch.zeros_like(rgb))  # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if rgb.is_cuda:
        mask = mask.cuda()

    rgb = (1.055 * ((rgb + 1e-12) ** (1. / 2.4)) - 0.055) * mask + 12.92 * rgb * (1 - mask)

    return rgb


def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    if xyz.is_cuda:
        sc = sc.cuda()

    xyz_scale = xyz / sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if xyz_scale.is_cuda:
        mask = mask.cuda()

    xyz_int = xyz_scale ** (1 / 3.) * mask + (7.787 * xyz_scale + 16. / 116.) * (1 - mask)

    L = 116. * xyz_int[:, 1, :, :] - 16.
    a = 500. * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
    b = 200. * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])
    out = torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)

    return out


def lab2xyz(lab):
    y_int = (lab[:, 0, :, :] + 16.) / 116.
    x_int = (lab[:, 1, :, :] / 500.) + y_int
    z_int = y_int - (lab[:, 2, :, :] / 200.)

    if z_int.is_cuda:
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if out.is_cuda:
        mask = mask.cuda()

    out = (out ** 3.) * mask + (out - 16. / 116.) / 7.787 * (1 - mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    sc = sc.to(out.device)

    out = out * sc
    return out


def lab2rgb_tensor(lab_tensor):
    """
    Parameter:
        lab_tensor : a 4D torch.Tensor instance, shape = (B, C, H, W)
        Accepted range for LAB: L = [0, 1] and AB = [-1, 1]
        network_opt : contains information needs for converting
    Return
        rgb_tensor : the input tensor in LAB color space, shape = (B, C, H, W)
        Output range : [0, 1]
    Explanation:
        convert the input tensors from RGB to LAB
    """
    l_tmp = lab_tensor[:, [0], :, :]
    a_tmp = lab_tensor[:, [1], :, :]
    b_tmp = lab_tensor[:, [2], :, :]
    l_tmp_d = l_tmp
    if color_output_mode == "A*L*B*L":
        if l_tmp.is_cuda:
            l_tmp_d = torch.max(torch.Tensor((0.01,)).cuda(), l_tmp)
        else:
            l_tmp_d = torch.max(torch.Tensor((0.01,)), l_tmp)
        a_tmp /= l_tmp_d
        b_tmp /= l_tmp_d
    l_tmp = l_tmp_d * l_norm + l_center
    a_tmp = a_tmp * a_norm + a_center
    b_tmp = b_tmp * b_norm + b_center

    lab_tmp = torch.cat((l_tmp, a_tmp, b_tmp), dim=1)
    return xyz2rgb(lab2xyz(lab_tmp))


def rgb2lab_tensor(rgb_tensor):
    """
        Parameter:
            rgb_tensor : a 4D torch.Tensor instance, shape = (B, C, H, W)
        Return
            lab_tensor : the input tensor in RGB color space, shape = (B, C, H, W)
        Explanation:
            convert the input tensors from LAB to RGB
        """
    return xyz2lab(rgb2xyz(rgb_tensor))


def downsample_PIL(rgb_img, scale):
    """
    :param rgb_img: input image is a NumPy array, [0, 1], RGB, HWC
    :param scale: the scaling factor
    :return: a NumPy array, [0, 1], np.float32, RGB, HWC
    """
    inp_rgb = Image.fromarray(np.uint8(rgb_img * 255.))
    transformed = np.array(inp_rgb.resize((int(inp_rgb.size[0] * scale), int(inp_rgb.size[1] * scale)),
                                           resample=PIL.Image.BICUBIC), dtype=np.float32)
    transformed /= 255.
    return transformed


####################
# metric
####################
def psnr(img1, img2):
    assert img1.dtype == img2.dtype == np.uint8, 'np.uint8 is supposed.'
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

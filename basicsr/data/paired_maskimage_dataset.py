import os

from torch.utils import data as data
from torchvision.transforms.functional import normalize
import cv2
from basicsr.data.data_util import mask_guide_paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr, imresize
from basicsr.utils.registry import DATASET_REGISTRY
import numpy as np

@DATASET_REGISTRY.register()
class PairedMASKDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_mask (str): Data root path for mask
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']  # disk, lmdb ,...
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.mask_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_mask'], opt['dataroot_lq']
        self.gt_lr_folder = opt['dataroot_gt_lr']
        self.avg_ct_folder = opt['dataroot_avg_ct']
        self.avg_ct_dic = self.create_avg_ct_dic()

        self.rgb = opt.get("rgb", False)

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            # self.paths是一个列表，列表的每一个元素是一个字典，字典有4项，分别是lq_path:lq_image_path, mask_path: mask_image_path, gt_path:gt_image_path, gt_lr_path:gt_lr_image_path
            self.paths = mask_guide_paired_paths_from_folder([self.lq_folder, self.mask_folder, self.gt_folder, self.gt_lr_folder], ['lq', 'mask', 'gt', 'gt_lr'], self.filename_tmpl)



    def __getitem__(self, index):
        # 可以对类的实例p进行索引，例如p[index]，这个方法定义了如何取数据
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)  # 用opencv加载二进制图片
        if not self.rgb:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = np.expand_dims(img_gt, 2)

        gt_lr_path = self.paths[index]['gt_lr_path']
        img_bytes = self.file_client.get(gt_lr_path, 'gt_lr')
        img_gt_lr = imfrombytes(img_bytes, float32=True)  # 用opencv加载二进制图片
        if not self.rgb:
            img_gt_lr = cv2.cvtColor(img_gt_lr, cv2.COLOR_BGR2GRAY)
            img_gt_lr = np.expand_dims(img_gt_lr, 2)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        if not self.rgb:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.expand_dims(img_lq, 2)

        ct_num1 = os.path.basename(lq_path)
        ct_num1 = os.path.splitext(ct_num1)[0]
        ct_num1 = ct_num1.split("_")[0]
        avg_ct_path = self.avg_ct_dic[ct_num1]
        img_bytes = self.file_client.get(avg_ct_path, 'avg_ct')
        img_avg_ct = imfrombytes(img_bytes, float32=True)
        if not self.rgb:
            img_avg_ct = cv2.cvtColor(img_avg_ct, cv2.COLOR_BGR2GRAY)
            img_avg_ct = np.expand_dims(img_avg_ct, 2)

        mask_path = self.paths[index]['mask_path']
        img_bytes = self.file_client.get(mask_path, 'mask')
        img_mask = imfrombytes(img_bytes, float32=True)  # 用opencv加载二进制图片
        if not self.rgb:
            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            img_mask = np.expand_dims(img_mask, 2)
        # w_gt, w_lq = img_gt.shape[0], img_lq.shape[0]
        # s = w_gt // w_lq
        # img_mask = imresize(img_mask, 1 / s)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, [img_gt_lr, img_mask, img_avg_ct, img_lq] = paired_random_crop(img_gt, [img_gt_lr, img_mask, img_avg_ct, img_lq], gt_size, scale, gt_path)  # 每一对图片只保留一个patch

            # flip, rotation
            img_gt, img_gt_lr, img_mask, img_avg_ct, img_lq = augment([img_gt, img_gt_lr, img_mask, img_avg_ct, img_lq], self.opt['use_flip'], self.opt['use_rot'])  # 对每一对图片进行随机旋转、翻转

        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_gt_lr = rgb2ycbcr(img_gt_lr, y_only=True)[..., None]
            img_mask = rgb2ycbcr(img_mask, y_only=True)[..., None]
            img_avg_ct = rgb2ycbcr(img_avg_ct, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_gt_lr, img_mask, img_avg_ct, img_lq = img2tensor([img_gt, img_gt_lr, img_mask, img_avg_ct, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_mask, self.mean, self.std, inplace=True)
            normalize(img_avg_ct, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_gt_lr, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'avg_ct': img_avg_ct, 'mask': img_mask, 'gt': img_gt, 'gt_lr': img_gt_lr, 'lq_path': lq_path, 'avg_ct_path': img_avg_ct, 'mask_path': mask_path, 'gt_path': gt_path, 'gt_lr_path': gt_lr_path}

    def __len__(self):
        return len(self.paths)


    def create_avg_ct_dic(self):
        avg_ct_dic = {}
        avg_names = os.listdir(self.avg_ct_folder)
        for avg_name in avg_names:
            num = os.path.splitext(avg_name)[0]
            avg_ct_dic[num] = os.path.join(self.avg_ct_folder, avg_name)
        return avg_ct_dic



@DATASET_REGISTRY.register()
class PairedMASKDataset_wo_avgct(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_mask (str): Data root path for mask
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']  # disk, lmdb ,...
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.mask_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_mask'], opt['dataroot_lq']
        self.gt_lr_folder = opt['dataroot_gt_lr']
        # self.avg_ct_folder = opt['dataroot_avg_ct']
        # self.avg_ct_dic = self.create_avg_ct_dic()

        self.rgb = opt.get("rgb", False)

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            # self.paths是一个列表，列表的每一个元素是一个字典，字典有4项，分别是lq_path:lq_image_path, mask_path: mask_image_path, gt_path:gt_image_path, gt_lr_path:gt_lr_image_path
            self.paths = mask_guide_paired_paths_from_folder([self.lq_folder, self.mask_folder, self.gt_folder, self.gt_lr_folder], ['lq', 'mask', 'gt', 'gt_lr'], self.filename_tmpl)



    def __getitem__(self, index):
        # 可以对类的实例p进行索引，例如p[index]，这个方法定义了如何取数据
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)  # 用opencv加载二进制图片
        if not self.rgb:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = np.expand_dims(img_gt, 2)

        gt_lr_path = self.paths[index]['gt_lr_path']
        img_bytes = self.file_client.get(gt_lr_path, 'gt_lr')
        img_gt_lr = imfrombytes(img_bytes, float32=True)  # 用opencv加载二进制图片
        if not self.rgb:
            img_gt_lr = cv2.cvtColor(img_gt_lr, cv2.COLOR_BGR2GRAY)
            img_gt_lr = np.expand_dims(img_gt_lr, 2)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        if not self.rgb:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.expand_dims(img_lq, 2)

        # ct_num1 = os.path.basename(lq_path)
        # ct_num1 = os.path.splitext(ct_num1)[0]
        # ct_num1 = ct_num1.split("_")[0]
        # avg_ct_path = self.avg_ct_dic[ct_num1]
        # img_bytes = self.file_client.get(avg_ct_path, 'avg_ct')
        # img_avg_ct = imfrombytes(img_bytes, float32=True)
        # if not self.rgb:
        #     img_avg_ct = cv2.cvtColor(img_avg_ct, cv2.COLOR_BGR2GRAY)
        #     img_avg_ct = np.expand_dims(img_avg_ct, 2)

        mask_path = self.paths[index]['mask_path']
        img_bytes = self.file_client.get(mask_path, 'mask')
        img_mask = imfrombytes(img_bytes, float32=True)  # 用opencv加载二进制图片
        if not self.rgb:
            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            img_mask = np.expand_dims(img_mask, 2)
        # w_gt, w_lq = img_gt.shape[0], img_lq.shape[0]
        # s = w_gt // w_lq
        # img_mask = imresize(img_mask, 1 / s)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, [img_gt_lr, img_mask, img_lq] = paired_random_crop(img_gt, [img_gt_lr, img_mask, img_lq], gt_size, scale, gt_path)  # 每一对图片只保留一个patch

            # flip, rotation
            img_gt, img_gt_lr, img_mask, img_lq = augment([img_gt, img_gt_lr, img_mask, img_lq], self.opt['use_flip'], self.opt['use_rot'])  # 对每一对图片进行随机旋转、翻转

        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_gt_lr = rgb2ycbcr(img_gt_lr, y_only=True)[..., None]
            img_mask = rgb2ycbcr(img_mask, y_only=True)[..., None]
            # img_avg_ct = rgb2ycbcr(img_avg_ct, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_gt_lr, img_mask, img_lq = img2tensor([img_gt, img_gt_lr, img_mask, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_mask, self.mean, self.std, inplace=True)
            # normalize(img_avg_ct, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_gt_lr, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'mask': img_mask, 'gt': img_gt, 'gt_lr': img_gt_lr, 'lq_path': lq_path, 'mask_path': mask_path, 'gt_path': gt_path, 'gt_lr_path': gt_lr_path}

    def __len__(self):
        return len(self.paths)

from scripts.metrics.calculate_psnr_ssim import main3
import os
import numpy as np
from multiprocessing import Process


def sort_imgs(name):
    if "." in name:
        basename = os.path.splitext(name)[0]
    num1, num2 = basename.split("_")
    return int(num1) * 10000 + int(num2)


def _get_psnr_and_ssim_for_bicubic(gts_dir, restored_dir, scale):
    gts = os.listdir(gts_dir)
    restoreds = os.listdir(restored_dir)
    gts = sorted(gts, key=sort_imgs)
    restoreds = sorted(restoreds, key=sort_imgs)
    gts = [os.path.join(gts_dir, i) for i in gts]
    restoreds = [os.path.join(restored_dir, j) for j in restoreds]
    psnr, ssim = main3(gts, restoreds, test_y_channel=True, crop_border=scale)
    print(f"{restored_dir} x{scale} \n PSNR AVG: {np.mean(psnr)} PSNR STD: {np.std(psnr)} \n SSIM AVG: {np.mean(ssim)} SSIM STD: {np.std(ssim)}")

def get_psnr_and_ssim_for_bicubic():
    # 在172.17.27.170运行

    bicubic_root_dir_3dircadb_x2 = "/home/zhiyi/data/3dircadb/img/lr_ld_bicubic/x2/test"
    bicubic_root_dir_3dircadb_x4 = "/home/zhiyi/data/3dircadb/img/lr_ld_bicubic/x4/test"
    bicubic_root_dir_pancreas_x2 = "/home/zhiyi/data/pancreas/img/lr_ld_bicubic/x2/test"
    bicubic_root_dir_pancreas_x4 = "/home/zhiyi/data/pancreas/img/lr_ld_bicubic/x4/test"

    gt_dir_3dircadb = "/home/zhiyi/data/3dircadb/img/hr_nd/test"
    gt_dir_pancreas = "/home/zhiyi/data/pancreas/img/hr_nd/test"

    tasks = []
    tasks.append(
        Process(target=_get_psnr_and_ssim_for_bicubic, args=(gt_dir_3dircadb, bicubic_root_dir_3dircadb_x2, 2))
    )
    tasks.append(
        Process(target=_get_psnr_and_ssim_for_bicubic, args=(gt_dir_3dircadb, bicubic_root_dir_3dircadb_x4, 4))
    )
    tasks.append(
        Process(target=_get_psnr_and_ssim_for_bicubic, args=(gt_dir_pancreas, bicubic_root_dir_pancreas_x2, 2))
    )
    tasks.append(
        Process(target=_get_psnr_and_ssim_for_bicubic, args=(gt_dir_pancreas, bicubic_root_dir_pancreas_x4, 4))
    )

    for task in tasks:
        task.start()

get_psnr_and_ssim_for_bicubic()
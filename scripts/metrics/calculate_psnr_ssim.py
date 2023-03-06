import argparse
import glob

import cv2
import numpy as np
from os import path as osp
import os
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import scandir
from basicsr.utils.matlab_functions import bgr2ycbcr
import csv


def main(args, save_txt_path=None, flag=cv2.IMREAD_UNCHANGED):
    """Calculate PSNR and SSIM for images.
    """
    psnr_all = []
    ssim_all = []
    img_list_gt = sorted(list(scandir(args.gt, recursive=True, full_path=True)))
    img_list_restored = sorted(list(scandir(args.restored, recursive=True, full_path=True)))

    if args.test_y_channel:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, flag).astype(np.float32) / 255.
        if len(img_gt.shape) == 3:
            img_gt = img_gt[:, :, 0]
        if args.suffix == '':
            img_path_restored = img_list_restored[i]
        else:
            img_path_restored = osp.join(args.restored, basename + args.suffix + ext)
        img_restored = cv2.imread(img_path_restored, flag).astype(np.float32) / 255.

        if args.correct_mean_var:
            mean_l = []
            std_l = []
            for j in range(3):
                mean_l.append(np.mean(img_gt[:, :, j]))
                std_l.append(np.std(img_gt[:, :, j]))
            for j in range(3):
                # correct twice
                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

        if args.test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # calculate PSNR and SSIM
        psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        print(f'{i + 1:3d}: {basename:25}. \tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}')
        # 原来是if not psnr
        if psnr:
            psnr_all.append(psnr)
        ssim_all.append(ssim)
    if save_txt_path is not None:
        with open(save_txt_path, "w") as file:
            print(args.gt, file=file)
            print(args.restored, file=file)
            print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f} dB, SSIM: {sum(ssim_all) / len(ssim_all):.6f}',
                  file=file)
    else:
        print(args.gt)
        print(args.restored)
        print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f} dB, SSIM: {sum(ssim_all) / len(ssim_all):.6f}')


def crop_circle(img):
    w, h = img.shape[:2]
    wc, hc = w // 2, h // 2
    r = min(wc, hc)
    for i in range(w):
        for j in range(h):
            if (i-wc) ** 2 + (j-hc) ** 2 > r ** 2:
                img[i, j] = 0
    return img

def main2(gts,
          restoreds,
          test_y_channel=False,
          suffix="",
          correct_mean_var=False,
          crop_border=0,
          txt_save_paths=None,
          circle=False):
    """Calculate PSNR and SSIM for images.
    """

    assert type(gts) == type(restoreds), "输入类型不符，全为列表或全为字符串"
    if isinstance(gts, str):
        gts = [gts]
        restoreds = [restoreds]


    if txt_save_paths is None:
        txt_save_paths = []
        for _ in range(len(gts)):
            txt_save_paths.append(None)

    for gt, restored, txt_save_path in zip(gts, restoreds, txt_save_paths):

        psnr_all = []
        ssim_all = []
        if not os.path.isdir(restored):
            continue
        img_list_gt = sorted(list(scandir(gt, recursive=True, full_path=True)))
        img_list_restored = sorted(list(scandir(restored, recursive=True, full_path=True)))

        if test_y_channel:
            print('Testing Y channel.')
        else:
            print('Testing RGB channels.')

        for i, img_path in enumerate(img_list_gt):
            basename, ext = osp.splitext(osp.basename(img_path))
            img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            if len(img_gt.shape) == 3:
                img_gt = img_gt[:, :, 0]
            if suffix == '':
                img_path_restored = img_list_restored[i]
            else:
                img_path_restored = osp.join(restored, basename + suffix + ext)
            img_restored = cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

            if circle:
                img_gt = crop_circle(img_gt)
                img_restored = crop_circle(img_restored)

            if correct_mean_var:
                mean_l = []
                std_l = []
                for j in range(3):
                    mean_l.append(np.mean(img_gt[:, :, j]))
                    std_l.append(np.std(img_gt[:, :, j]))
                for j in range(3):
                    # correct twice
                    mean = np.mean(img_restored[:, :, j])
                    img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                    std = np.std(img_restored[:, :, j])
                    img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                    mean = np.mean(img_restored[:, :, j])
                    img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                    std = np.std(img_restored[:, :, j])
                    img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

            if test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
                img_gt = bgr2ycbcr(img_gt, y_only=True)
                img_restored = bgr2ycbcr(img_restored, y_only=True)

            # calculate PSNR and SSIM
            psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=crop_border, input_order='HWC')
            ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=crop_border, input_order='HWC')
            print(f'{i + 1:3d}: {basename:25}. \tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}')
            # 原来是if not psnr
            if psnr:
                psnr_all.append(psnr)
            ssim_all.append(ssim)
        if txt_save_path is not None:
            with open(txt_save_path, "w") as file:
                print(gt, file=file)
                print(restored, file=file)
                print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f} dB, SSIM: {sum(ssim_all) / len(ssim_all):.6f}',
                      file=file)
        else:
            print(gt)
            print(restored)
            print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f} dB, SSIM: {sum(ssim_all) / len(ssim_all):.6f}')
        return sum(psnr_all) / len(psnr_all), sum(ssim_all) / len(ssim_all)


def main3(gts: list,
          restoreds: list,
          test_y_channel=False,
          correct_mean_var=False,
          crop_border=0,
          circle=False):
    """
    Args:
        gts: gt文件路径
        restoreds: 测试图像路径
        test_y_channel:
        suffix:
        correct_mean_var:
        crop_border:
        circle: 是否只统计中心圆区域

    Returns:
        psnr_all, ssim_all
    """
    psnr_all = []
    ssim_all = []

    img_list_gt = gts
    img_list_restored = restoreds

    if test_y_channel:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        # if len(img_gt.shape) == 3:
        #     img_gt = img_gt[:, :, 0]

        img_path_restored = img_list_restored[i]
        img_restored = cv2.imread(img_path_restored, cv2.IMREAD_COLOR).astype(np.float32) / 255.

        if circle:
            img_gt = crop_circle(img_gt)
            img_restored = crop_circle(img_restored)

        if correct_mean_var:
            mean_l = []
            std_l = []
            for j in range(3):
                mean_l.append(np.mean(img_gt[:, :, j]))
                std_l.append(np.std(img_gt[:, :, j]))
            for j in range(3):
                # correct twice
                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

        if test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # calculate PSNR and SSIM
        # if img_gt.shape != img_restored.shape:
        #     print(img_path_restored, img_gt.shape, img_restored.shape)
        #     continue
        #     w1, h1 = img_gt.shape[:2]
        #     w2, h2 = img_restored.shape[:2]
        #     w = min(w1, w2)
        #     h = min(h1, h2)
        #     img_gt = img_gt[:w, :h]
        #     img_restored = img_restored[:w, :h]
        psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=crop_border, input_order='HWC')
        ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=crop_border, input_order='HWC')
        # print(f'{i + 1:3d}: {basename:25}. \tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}')
        # 原来是if not psnr
        if psnr:
            psnr_all.append(psnr)
        ssim_all.append(ssim)
        # print(os.path.basename(restoreds[0][:-4].split("_")[-1]))
    if len(psnr_all) > 0:
        # print(gts[0])
        # print(restoreds[0])
        print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f} dB, SSIM: {sum(ssim_all) / len(ssim_all):.6f}')
        # return sum(psnr_all) / len(psnr_all), np.std(psnr_all), sum(ssim_all) / len(ssim_all), np.std(ssim_all)
        return psnr_all, ssim_all
    else:
        return None, None


def test_denoise_paper():
    adnet_restored_dir = ["/mnt/c/data/去噪结果/adnet/luna16_test",
                          "/mnt/c/data/去噪结果/adnet/aapm_L067",
                          "/mnt/c/data/去噪结果/adnet/aapm_L096"]
    bm3d_restored_dir = ["/mnt/c/data/去噪结果/BM3D/luna16_test",
                         "/mnt/c/data/去噪结果/BM3D/aapm_L067",
                         "/mnt/c/data/去噪结果/BM3D/aapm_L096"]
    ctformer_restored_dir = ["/mnt/c/data/去噪结果/ctformer/luna16_test",
                             "/mnt/c/data/去噪结果/ctformer/aapm_L067",
                             "/mnt/c/data/去噪结果/ctformer/aapm_L096"]
    dugan_restored_dir = ["/mnt/c/data/去噪结果/dugan/luna16_test",
                          "/mnt/c/data/去噪结果/dugan/aapm_L067",
                          "/mnt/c/data/去噪结果/dugan/aapm_L096"]
    qae_restored_dir = ["/mnt/c/data/去噪结果/qae/luna16_test",
                        "/mnt/c/data/去噪结果/qae/aapm_L067",
                        "/mnt/c/data/去噪结果/qae/aapm_L096"]
    deamnet_restored_dir = ["/mnt/c/data/去噪结果/deamnet/luna16_test",
                        "/mnt/c/data/去噪结果/deamnet/aapm_L067",
                        "/mnt/c/data/去噪结果/deamnet/aapm_L096"]
    redcnn_restored_dir = ["/mnt/c/data/去噪结果/redcnn/luna16_test",
                           "/mnt/c/data/去噪结果/redcnn/aapm_L067",
                           "/mnt/c/data/去噪结果/redcnn/aapm_L096"]
    redcnn_mean_restored_dir = ["/mnt/c/data/去噪结果/proposed/dual_domain_mean/luna16_test"]
    redcnn_sino_restored_dir = ["/mnt/c/data/去噪结果/proposed/redcnn_dwt_out/luna16_test",
                                        "/mnt/c/data/去噪结果/proposed/redcnn_dwt_out/aapm_L067",
                                        "/mnt/c/data/去噪结果/proposed/redcnn_dwt_out/aapm_L096"]
    ifm_redcnn_restored_dir = ["/mnt/c/data/去噪结果/proposed/irn_out/luna16_test",
                                 "/mnt/c/data/去噪结果/proposed/irn_out/aapm_L067",
                                 "/mnt/c/data/去噪结果/proposed/irn_out/aapm_L096"]
    # ifm_ircnn_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/ircnn/luna16_test",
    #                            "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L067",
    #                            "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L096"]
    # ifm_memnet_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/memnet/luna16_test",
    #                            "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L067",
    #                            "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L096"]
    # ifm_dncnn_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/dncnn/luna16_test",
    #                            "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L067",
    #                            "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L096"]
    # ifm_ridnet_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/ridnet/luna16_test",
    #                            "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L067",
    #                            "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L096"]
    proposed_3dresunet_mean_restored_dir = ["/mnt/c/data/去噪结果/proposed/3dresunet/luna16_test/mean",
                                            "/mnt/c/data/去噪结果/proposed/3dresunet/aapm_L067/mean",
                                            "/mnt/c/data/去噪结果/proposed/3dgcnunet/aapm_L096/mean"]
    proposed_3dresunet_best_restored_dir = ["/mnt/c/data/去噪结果/proposed/3dresunet/luna16_test/best",
                                            "/mnt/c/data/去噪结果/proposed/3dresunet/aapm_L067/best",
                                            "/mnt/c/data/去噪结果/proposed/3dgcnunet/aapm_L096/best"]
    proposed_3dgcnunet_only_mean_restored_dir = ["/mnt/c/data/去噪结果/proposed/3dgcnunet_only_out/luna16_test/mean",
                                            "/mnt/c/data/去噪结果/proposed/3dgcnunet_only_out/aapm_L067/mean",
                                            "/mnt/c/data/去噪结果/proposed/3dgcnunet_only_out/aapm_L096/mean"]
    proposed_3dgcnunet_only_best_restored_dir = ["/mnt/c/data/去噪结果/proposed/3dgcnunet_only_out/luna16_test/best",
                                            "/mnt/c/data/去噪结果/proposed/3dgcnunet_only_out/aapm_L067/best",
                                            "/mnt/c/data/去噪结果/proposed/3dgcnunet_only_out/aapm_L096/best"]
    proposed_3dgcnunet_mean_restored_dir = ["/mnt/c/data/去噪结果/proposed/3dgcnunet_out/luna16_test/mean",
                                            "/mnt/c/data/去噪结果/proposed/3dgcnunet_out/aapm_L067/mean",
                                            "/mnt/c/data/去噪结果/proposed/3dgcnunet_out/aapm_L096/mean"]
    proposed_3dgcnunet_best_restored_dir = ["/mnt/c/data/去噪结果/proposed/3dgcnunet_out/luna16_test/best",
                                            "/mnt/c/data/去噪结果/proposed/3dgcnunet_out/aapm_L067/best",
                                            "/mnt/c/data/去噪结果/proposed/3dgcnunet_out/aapm_L096/best"]

    dncnn_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/dncnn/luna16_test/img_branch",
                          "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L067/img_branch",
                          "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L096/img_branch"]
    dncnn_sino_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/dncnn/luna16_test/sino_branch",
                               "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L067/sino_branch",
                               "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L096/sino_branch"]
    ifm_dncnn_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/dncnn/luna16_test/ifm",
                                       "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L067/ifm",
                                       "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L096/ifm"]
    proposed_3dgcnunet_dncnn_best_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/dncnn/luna16_test/3dgcnunet/best",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L067/3dgcnunet/best",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L096/3dgcnunet/best"]
    memnet_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/memnet/luna16_test/img_branch",
                           "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L067/img_branch",
                           "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L096/img_branch"]
    memnet_sino_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/memnet/luna16_test/sino_branch",
                                "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L067/sino_branch",
                                "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L096/sino_branch"]
    ifm_memnet_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/memnet/luna16_test/ifm",
                                        "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L067/ifm",
                                        "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L096/ifm"]
    proposed_3dgcnunet_memnet_best_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/memnet/luna16_test/3dgcnunet/best",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L067/3dgcnunet/best",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L096/3dgcnunet/best"]
    ridnet_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/ridnet/luna16_test/img_branch",
                           "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L067/img_branch",
                           "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L096/img_branch"]
    ridnet_sino_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/ridnet/luna16_test/sino_branch",
                                "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L067/sino_branch",
                                "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L096/sino_branch"]
    ifm_ridnet_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/ridnet/luna16_test/ifm",
                               "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L067/ifm",
                               "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L096/ifm"]
    proposed_3dgcnunet_ridnet_best_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/ridnet/luna16_test/3dgcnunet/best",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L067/3dgcnunet/best",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L096/3dgcnunet/best"]
    ircnn_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/ircnn/luna16_test/img_branch",
                          "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L067/img_branch",
                          "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L096/img_branch"]
    ircnn_sino_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/ircnn/luna16_test/sino_branch",
                               "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L067/sino_branch",
                               "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L096/sino_branch"]
    ifm_ircnn_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/ircnn/luna16_test/ifm",
                                       "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L067/ifm",
                                       "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L096/ifm"]
    proposed_3dgcnunet_ircnn_best_restored_dir = ["/mnt/c/data/去噪结果/模型兼容性实验/ircnn/luna16_test/3dgcnunet/best",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L067/3dgcnunet/best",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L096/3dgcnunet/best"]
    gt_dir = ["/mnt/c/data/LUNA16/0/luna16up/nd/test",
              "/mnt/c/data/aapm/L067/nd",
              "/mnt/c/data/aapm/L096/nd"]
    proposed_3dgcnunet_gt_dir = ["/mnt/c/data/LUNA16/0/luna16up/nd/test",
                                 "/mnt/c/data/aapm/L067_3dgcnunet/nd",
                                 "/mnt/c/data/aapm/L096_3dgcnunet/nd"]

    adnet_txt_save_path = ["/mnt/c/data/去噪结果/adnet/luna16_test.log",
                           "/mnt/c/data/去噪结果/adnet/aapm_L067.log",
                           "/mnt/c/data/去噪结果/adnet/aapm_L096.log"]
    bm3d_txt_save_path = ["/mnt/c/data/去噪结果/BM3D/luna16_test.log",
                           "/mnt/c/data/去噪结果/BM3D/aapm_L067.log",
                           "/mnt/c/data/去噪结果/BM3D/aapm_L096.log"]
    ctformer_txt_save_path = ["/mnt/c/data/去噪结果/ctformer/luna16_test.log",
                           "/mnt/c/data/去噪结果/ctformer/aapm_L067.log",
                           "/mnt/c/data/去噪结果/ctformer/aapm_L096.log"]
    dugan_txt_save_path = ["/mnt/c/data/去噪结果/dugan/luna16_test.log",
                           "/mnt/c/data/去噪结果/dugan/aapm_L067.log",
                           "/mnt/c/data/去噪结果/dugan/aapm_L096.log"]
    qae_txt_save_path = ["/mnt/c/data/去噪结果/qae/luna16_test.log",
                           "/mnt/c/data/去噪结果/qae/aapm_L067.log",
                           "/mnt/c/data/去噪结果/qae/aapm_L096.log"]
    deamnet_txt_save_path = ["/mnt/c/data/去噪结果/deamnet/luna16_test.log",
                           "/mnt/c/data/去噪结果/deamnet/aapm_L067.log",
                           "/mnt/c/data/去噪结果/deamnet/aapm_L096.log"]
    redcnn_txt_save_path = ["/mnt/c/data/去噪结果/redcnn/luna16_test.log",
                            "/mnt/c/data/去噪结果/redcnn/aapm_L067.log",
                            "/mnt/c/data/去噪结果/redcnn/aapm_L096.log"]
    redcnn_sino_txt_save_path = ["/mnt/c/data/去噪结果/proposed/redcnn_dwt_out/luna16_test.log",
                           "/mnt/c/data/去噪结果/proposed/redcnn_dwt_out/aapm_L067.log",
                           "/mnt/c/data/去噪结果/proposed/redcnn_dwt_out/aapm_L096.log"]
    redcnn_mean_txt_save_path = ["/mnt/c/data/去噪结果/proposed/dual_domain_mean/luna16_test.log"]
    ifm_redcnn_txt_save_path = ["/mnt/c/data/去噪结果/proposed/irn_out/luna16_test.log",
                            "/mnt/c/data/去噪结果/proposed/irn_out/aapm_L067.log",
                            "/mnt/c/data/去噪结果/proposed/irn_out/aapm_L096.log"]
    # ifm_ircnn_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/ircnn/luna16_test.log",
    #                            "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L067.log",
    #                            "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L096.log"]
    # ifm_dncnn_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/dncnn/luna16_test.log",
    #                                "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L067.log",
    #                                "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L096.log"]
    # ifm_memnet_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/memnet/luna16_test.log",
    #                                "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L067.log",
    #                                "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L096.log"]
    # ifm_ridnet_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/ridnet/luna16_test.log",
    #                                "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L067.log",
    #                                "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L096.log"]

    proposed_3dresunet_mean_txt_save_path = ["/mnt/c/data/去噪结果/proposed/3dresunet/luna16_test_mean.log",
                                             "/mnt/c/data/去噪结果/proposed/3dresunet/aapm_L067_mean.log",
                                             "/mnt/c/data/去噪结果/proposed/3dresunet/aapm_L096_mean.log"]
    proposed_3dresunet_best_txt_save_path = ["/mnt/c/data/去噪结果/proposed/3dresunet/luna16_test_best.log",
                                             "/mnt/c/data/去噪结果/proposed/3dresunet/aapm_L067_best.log",
                                             "/mnt/c/data/去噪结果/proposed/3dresunet/aapm_L096_best.log"]
    proposed_3dgcnunet_only_mean_txt_save_path = ["/mnt/c/data/去噪结果/proposed/3dgcnunet_only_out/luna16_test_mean.log",
                                             "/mnt/c/data/去噪结果/proposed/3dgcnunet_only_out/aapm_L067_mean.log",
                                             "/mnt/c/data/去噪结果/proposed/3dgcnunet_only_out/aapm_L096_mean.log"]
    proposed_3dgcnunet_only_best_txt_save_path = ["/mnt/c/data/去噪结果/proposed/3dgcnunet_only_out/luna16_test_best.log",
                                             "/mnt/c/data/去噪结果/proposed/3dgcnunet_only_out/aapm_L067_best.log",
                                             "/mnt/c/data/去噪结果/proposed/3dgcnunet_only_out/aapm_L096_best.log"]
    proposed_3dgcnunet_mean_txt_save_path = ["/mnt/c/data/去噪结果/proposed/3dgcnunet_only_out/luna16_test_mean.log",
                                             "/mnt/c/data/去噪结果/proposed/3dgcnunet_out/aapm_L067_mean.log",
                                             "/mnt/c/data/去噪结果/proposed/3dgcnunet_out/aapm_L096_mean.log"]
    proposed_3dgcnunet_best_txt_save_path = ["/mnt/c/data/去噪结果/proposed/3dgcnunet_out/luna16_test_best.log",
                                             "/mnt/c/data/去噪结果/proposed/3dgcnunet_out/aapm_L067_best.log",
                                             "/mnt/c/data/去噪结果/proposed/3dgcnunet_out/aapm_L096_best.log"]

    dncnn_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/dncnn/luna16_test/img_branch.log",
                           "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L067/img_branch.log",
                           "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L096/img_branch.log"]
    dncnn_sino_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/dncnn/luna16_test/sino_branch.log",
                                "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L067/sino_branch.log",
                                "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L096/sino_branch.log"]
    ifm_dncnn_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/dncnn/luna16_test/ifm.log",
                               "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L067/ifm.log",
                               "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L096/ifm.log"]
    proposed_3dgcnunet_dncnn_best_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/dncnn/luna16_test/3dgcnunet/best.log",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L067/3dgcnunet/best.log",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/dncnn/aapm_L096/3dgcnunet/best.log"]

    memnet_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/memnet/luna16_test/img_branch.log",
                            "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L067/img_branch.log",
                            "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L096/img_branch.log"]
    memnet_sino_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/memnet/luna16_test/sino_branch.log",
                                 "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L067/sino_branch.log",
                                 "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L096/sino_branch.log"]
    ifm_memnet_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/memnet/luna16_test/ifm.log",
                                "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L067/ifm.log",
                                "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L096/ifm.log"]
    proposed_3dgcnunet_memnet_best_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/memnet/luna16_test/3dgcnunet/best.log",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L067/3dgcnunet/best.log",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/memnet/aapm_L096/3dgcnunet/best.log"]

    ircnn_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/ircnn/luna16_test/img_branch.log",
                           "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L067/img_branch.log",
                           "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L096/img_branch.log"]
    ircnn_sino_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/ircnn/luna16_test/sino_branch.log",
                                "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L067/sino_branch.log",
                                "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L096/sino_branch.log"]
    ifm_ircnn_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/ircnn/luna16_test/ifm.log",
                               "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L067/ifm.log",
                               "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L096/ifm.log"]
    proposed_3dgcnunet_ircnn_best_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/ircnn/luna16_test/3dgcnunet/best.log",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L067/3dgcnunet/best.log",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/ircnn/aapm_L096/3dgcnunet/best.log"]

    ridnet_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/ridnet/luna16_test/img_branch.log",
                            "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L067/img_branch.log",
                            "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L096/img_branch.log"]
    ridnet_sino_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/ridnet/luna16_test/sino_branch.log",
                                 "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L067/sino_branch.log",
                                 "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L096/sino_branch.log"]
    ifm_ridnet_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/ridnet/luna16_test/ifm.log",
                                "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L067/ifm.log",
                                "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L096/ifm.log"]
    proposed_3dgcnunet_ridnet_best_txt_save_path = ["/mnt/c/data/去噪结果/模型兼容性实验/ridnet/luna16_test/3dgcnunet/best.log",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L067/3dgcnunet/best.log",
                                                   "/mnt/c/data/去噪结果/模型兼容性实验/ridnet/aapm_L096/3dgcnunet/best.log"]

    config = {
        "adnet": {"gts": gt_dir,
                  "restoreds": adnet_restored_dir,
                  "txt_save_paths": adnet_txt_save_path},
        "BM3D": {"gts": gt_dir,
                 "restoreds": bm3d_restored_dir,
                 "txt_save_paths": bm3d_txt_save_path},
        "ctformer": {"gts": gt_dir,
                     "restoreds": ctformer_restored_dir,
                     "txt_save_paths": ctformer_txt_save_path},
        "dugan": {"gts": gt_dir,
                  "restoreds": dugan_restored_dir,
                  "txt_save_paths": dugan_txt_save_path},
        "qae": {"gts": gt_dir,
                "restoreds": qae_restored_dir,
                "txt_save_paths": qae_txt_save_path},
        "deamnet": {"gts": gt_dir,
                "restoreds": deamnet_restored_dir,
                "txt_save_paths": deamnet_txt_save_path},
        "redcnn_img": {"gts": gt_dir,
                       "restoreds": redcnn_restored_dir,
                       "txt_save_paths": redcnn_txt_save_path},
        "redcnn_sino": {"gts": gt_dir,
                        "restoreds": redcnn_sino_restored_dir,
                        "txt_save_paths": redcnn_sino_txt_save_path},
        "redcnn_mean": {"gts": gt_dir,
                        "restoreds": redcnn_mean_restored_dir,
                        "txt_save_paths": redcnn_mean_txt_save_path},
        "ifm_redcnn": {"gts": gt_dir,
                       "restoreds": ifm_redcnn_restored_dir,
                       "txt_save_paths": ifm_redcnn_txt_save_path},
        "3dresunet_mean": {"gts": proposed_3dgcnunet_gt_dir,
                           "restoreds": proposed_3dresunet_mean_restored_dir,
                           "txt_save_paths": proposed_3dresunet_mean_txt_save_path},
        "3dresunet_best": {"gts": proposed_3dgcnunet_gt_dir,
                           "restoreds": proposed_3dresunet_best_restored_dir,
                           "txt_save_paths": proposed_3dresunet_best_txt_save_path},
        "3dgcnunet_redcnn_only_mean": {"gts": proposed_3dgcnunet_gt_dir,
                                       "restoreds": proposed_3dgcnunet_only_mean_restored_dir,
                                       "txt_save_paths": proposed_3dgcnunet_only_mean_txt_save_path},
        "3dgcnunet_redcnn_only_best": {"gts": proposed_3dgcnunet_gt_dir,
                                       "restoreds": proposed_3dgcnunet_only_best_restored_dir,
                                       "txt_save_paths": proposed_3dgcnunet_only_best_txt_save_path},
        "3dgcnunet_redcnn_mean": {"gts": proposed_3dgcnunet_gt_dir,
                                  "restoreds": proposed_3dgcnunet_mean_restored_dir,
                                  "txt_save_paths": proposed_3dgcnunet_mean_txt_save_path},
        "3dgcnunet_redcnn_best": {"gts": proposed_3dgcnunet_gt_dir,
                                  "restoreds": proposed_3dgcnunet_best_restored_dir,
                                  "txt_save_paths": proposed_3dgcnunet_best_txt_save_path},
        "dncnn_img": {"gts": gt_dir,
                      "restoreds": dncnn_restored_dir,
                      "txt_save_paths": dncnn_txt_save_path},
        "dncnn_sino": {"gts": gt_dir,
                       "restoreds": dncnn_sino_restored_dir,
                       "txt_save_paths": dncnn_sino_txt_save_path},
        "ifm_dncnn": {"gts": gt_dir,
                      "restoreds": ifm_dncnn_restored_dir,
                      "txt_save_paths": ifm_dncnn_txt_save_path},
        "3dgcnunet_dncnn_best": {"gts": proposed_3dgcnunet_gt_dir,
                                 "restoreds": proposed_3dgcnunet_dncnn_best_restored_dir,
                                 "txt_save_paths": proposed_3dgcnunet_dncnn_best_txt_save_path},
        "memnet_img": {"gts": gt_dir,
                       "restoreds": memnet_restored_dir,
                       "txt_save_paths": memnet_txt_save_path},
        "memnet_sino": {"gts": gt_dir,
                        "restoreds": memnet_sino_restored_dir,
                        "txt_save_paths": memnet_sino_txt_save_path},
        "ifm_memnet": {"gts": gt_dir,
                       "restoreds": ifm_memnet_restored_dir,
                       "txt_save_paths": ifm_memnet_txt_save_path},
        "3dgcnunet_memnet_best": {"gts": proposed_3dgcnunet_gt_dir,
                                 "restoreds": proposed_3dgcnunet_memnet_best_restored_dir,
                                 "txt_save_paths": proposed_3dgcnunet_memnet_best_txt_save_path},
        "ircnn_img": {"gts": gt_dir,
                      "restoreds": ircnn_restored_dir,
                      "txt_save_paths": ircnn_txt_save_path},
        "ircnn_sino": {"gts": gt_dir,
                       "restoreds": ircnn_sino_restored_dir,
                       "txt_save_paths": ircnn_sino_txt_save_path},
        "ifm_ircnn": {"gts": gt_dir,
                      "restoreds": ifm_ircnn_restored_dir,
                      "txt_save_paths": ifm_ircnn_txt_save_path},
        "3dgcnunet_ircnn_best": {"gts": proposed_3dgcnunet_gt_dir,
                                 "restoreds": proposed_3dgcnunet_ircnn_best_restored_dir,
                                 "txt_save_paths": proposed_3dgcnunet_ircnn_best_txt_save_path},
        "ridnet_img": {"gts": gt_dir,
                       "restoreds": ridnet_restored_dir,
                       "txt_save_paths": ridnet_txt_save_path},
        "ridnet_sino": {"gts": gt_dir,
                        "restoreds": ridnet_sino_restored_dir,
                        "txt_save_paths": ridnet_sino_txt_save_path},
        "ifm_ridnet": {"gts": gt_dir,
                       "restoreds": ifm_ridnet_restored_dir,
                       "txt_save_paths": ifm_ridnet_txt_save_path},
        "3dgcnunet_ridnet_best": {"gts": proposed_3dgcnunet_gt_dir,
                                 "restoreds": proposed_3dgcnunet_ridnet_best_restored_dir,
                                 "txt_save_paths": proposed_3dgcnunet_ridnet_best_txt_save_path}
    }

    # main2(**config["3dresunet_mean"])
    # main2(**config["3dresunet_best"])
    # main2(**config["irn"])
    # main2(**config["3dgcnunet_only_mean"])
    # main2(**config["3dgcnunet_only_best"])
    # main2(**config["dncnn_img"])
    # main2(**config["dncnn_sino"], circle=True)
    main2(**config["deamnet"])
    # main2(**config["memnet_sino"], circle=True)


def test_denoise_paper_multi_level():
    bm3d_restored_dir = [f"/home/zhiyi/denoise_result/bm3d/{i}" for i in range(1, 11)]
    adnet_restored_dir = [f"/home/zhiyi/denoise_result/adnet/{i}" for i in range(1, 11)]
    ctformer_restored_dir = [f"/home/zhiyi/denoise_result/ctformer/{i}" for i in range(1, 11)]
    dugan_restored_dir = [f"/home/zhiyi/denoise_result/dugan/{i}" for i in range(1, 11)]
    qae_restored_dir = [f"/home/zhiyi/denoise_result/qae/{i}" for i in range(1, 11)]
    proposed_redcnn_img_restored_dir = [f"/home/zhiyi/denoise_result/proposed/img_branch/{i}" for i in range(1, 11)]
    proposed_redcnn_sino_restored_dir = [f"/home/zhiyi/denoise_result/proposed/sino_branch/{i}" for i in range(1, 11)]
    proposed_ifm_restored_dir = [f"/home/zhiyi/denoise_result/proposed/ifm/{i}" for i in range(1, 11)]
    proposed_3dgcnunet_mean_restored_dir = [f"/home/zhiyi/denoise_result/proposed/3dgcnunet/{i}/mean" for i in range(1, 11)]
    proposed_3dgcnunet_best_restored_dir = [f"/home/zhiyi/denoise_result/proposed/3dgcnunet/{i}/best" for i in range(1, 11)]


    gt_dir = ["/home/zhiyi/data/medical/luna16up/nd/test"] * 10

    bm3d_txt_save_path = [f"/home/zhiyi/denoise_result/bm3d/{i}.log" for i in range(1, 11)]
    adnet_txt_save_path = ["/home/zhiyi/denoise_result/adnet/{i}.log" for i in range(1, 11)]
    ctformer_txt_save_path = [f"/home/zhiyi/denoise_result/ctformer/{i}.log" for i in range(1, 11)]
    dugan_txt_save_path = [f"/home/zhiyi/denoise_result/dugan/{i}.log" for i in range(1, 11)]
    qae_txt_save_path = [f"/home/zhiyi/denoise_result/qae/{i}.log" for i in range(1, 11)]
    proposed_redcnn_img_txt_save_path = [f"/home/zhiyi/denoise_result/proposed/img_branch/{i}.log" for i in range(1, 11)]
    proposed_redcnn_sino_txt_save_path = [f"/home/zhiyi/denoise_result/proposed/sino_branch/{i}.log" for i in range(1, 11)]
    proposed_ifm_txt_save_path = [f"/home/zhiyi/denoise_result/proposed/ifm/{i}.log" for i in range(1, 11)]
    proposed_3dgcnunet_mean_txt_save_path = [f"/home/zhiyi/denoise_result/proposed/3dgcnunet/mean/{i}.log" for i in range(1, 11)]
    proposed_3dgcnunet_best_txt_save_path = [f"/home/zhiyi/denoise_result/proposed/3dgcnunet/best/{i}.log" for i in range(1, 11)]


    config = {
        "adnet": {"gts": gt_dir, "restoreds": adnet_restored_dir, "txt_save_paths": adnet_txt_save_path},
        "BM3D": {"gts": gt_dir, "restoreds": bm3d_restored_dir, "txt_save_paths": bm3d_txt_save_path},
        "ctformer": {"gts": gt_dir, "restoreds": ctformer_restored_dir, "txt_save_paths": ctformer_txt_save_path},
        "dugan": {"gts": gt_dir, "restoreds": dugan_restored_dir, "txt_save_paths": dugan_txt_save_path},
        "qae": {"gts": gt_dir, "restoreds": qae_restored_dir, "txt_save_paths": qae_txt_save_path},
        "redcnn_img": {"gts": gt_dir, "restoreds": proposed_redcnn_img_restored_dir, "txt_save_paths": proposed_redcnn_img_txt_save_path},
        "redcnn_dwt": {"gts": gt_dir, "restoreds": proposed_redcnn_sino_restored_dir,
                       "txt_save_paths": proposed_redcnn_sino_txt_save_path},
        # "redcnn_mean": {"gts": gt_dir, "restoreds": redcnn_mean_restored_dir,
        #                 "txt_save_paths": redcnn_mean_txt_save_path},
        "ifm": {"gts": gt_dir, "restoreds": proposed_ifm_restored_dir, "txt_save_paths": proposed_ifm_txt_save_path},
        "3dgcnunet_mean": {"gts": gt_dir, "restoreds": proposed_3dgcnunet_mean_restored_dir,
                           "txt_save_paths": proposed_3dgcnunet_mean_txt_save_path},
        "3dgcnunet_best": {"gts": gt_dir, "restoreds": proposed_3dgcnunet_best_restored_dir,
                           "txt_save_paths": proposed_3dgcnunet_best_txt_save_path},
    }

    # main2(**config["dugan"])


def test_ctsr_paper_ablation(exp_dir=None, gt_dir="/home/zhiyi/data/medical/belly/hr_valid"):
    gts = os.listdir(gt_dir)
    gts = sorted(gts, key=lambda x: int(x[:-4].split("_")[-1]))
    gts = [os.path.join(gt_dir, i) for i in gts]
    vis_dir = os.path.join(exp_dir, "visualization")
    img_path_dic = {}
    for i in range(1, 51):
        img_path_dic[i*10000] = []
    img_names = os.listdir(vis_dir)
    img_names = sorted(img_names, key=lambda x: int(x.split("_")[-1]))
    for img_name in img_names:
        img_dir = os.path.join(vis_dir, img_name)
        img_iters = os.listdir(img_dir)
        img_iters = sorted(img_iters, key=lambda x: int(x[:-4].split("_")[-1]))
        for img_iter in img_iters:
            img_path = os.path.join(vis_dir, img_name, img_iter)
            iter_ = int(img_iter[:-4].split("_")[-1])
            img_path_dic[iter_].append(img_path)
    with open(os.path.join(exp_dir, "PSNR_SSIM.csv"), "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        for k, v in img_path_dic.items():
            psnr, ssim = main3(gts=gts, restoreds=v, test_y_channel=True, crop_border=8)
            writer.writerow([k, np.mean(psnr), np.mean(ssim)])


def test_ctsr_paper_multidegration(exp_dir, hr_val_dir, coronal_dir, sagittal_dir, csv_save_path):
    gt_dic = {
        47: hr_val_dir,
        260: sagittal_dir,
        174: coronal_dir
    }
    vis_dir = os.path.join(exp_dir, "visualization")
    restored_names = os.listdir(vis_dir)
    with open(csv_save_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        name = []
        psnrs = []
        ssims = []
        for restored_name in restored_names:
            restored_dir = os.path.join(vis_dir, restored_name)
            restoreds_ = os.listdir(restored_dir)
            restoreds = [os.path.join(restored_dir, i) for i in restoreds_]
            gt_dir = gt_dic[len(restoreds_)]
            gts_ = os.listdir(gt_dir)
            gts = [os.path.join(gt_dir, i) for i in gts_]
            psnr, ssim = main3(gts, restoreds, True, crop_border=4)
            name.append(restored_name)
            psnrs.append(np.mean(psnr))
            ssims.append(np.mean(ssim))
        res = ["net_name"]
        for i in name:
            res.append(i)
            res.append(None)
        w.writerow(res)
        res = ["ignn"]
        for a, b in zip(psnrs, ssims):
            res.append(a)
            res.append(b)
        w.writerow(res)

def test_ctsr_paper_for_std(flag1=False, flag2=False, flag3=False):
    # 测试所有实验的PSNR、SSIM及其标准差,保存
    # 在WSL2环境下运行,保存每张图的名字及指标,格式为.csv文件

    methods = ["bicubic", "edsr", "rcan", "esrgan", "myhan", "ignn", "seanet", "nlsa", "swinir", "ghasr"]

    # 1. 与其他方法的对比(多数据集)
    if flag1:
        restored_root_dir = "/mnt/d/GSTSR-related-data/results/zt_{}_x{}_ct/visualization/{}"
        gt_root_dir = "/mnt/c/data/测试数据集/{}/hr_valid"
        scales = [2, 4, 8]
        dataset_names = ["3d-ircadb", "pancreas", "tcia"]
        for scale in scales:
            csv_file = f"/mnt/c/Users/NEU_s/OneDrive/document/想法/CT超分/补充实验/统计分析/multi_dataset_x{scale}.csv"
            with open(csv_file, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                head = ["method"]
                for i in dataset_names:
                    head.append(i)
                    head.append("")
                    head.append("")
                    head.append("")
                sub_head = [""] + ["psnr_avg", "psnr_std", "ssim_avg", "ssim_std"] * len(dataset_names)
                w.writerow(head)
                w.writerow(sub_head)
                for method in methods:
                    line = [method]
                    for dataset_name in dataset_names:
                        gt_dir = gt_root_dir.format(dataset_name)
                        img_dir = restored_root_dir.format(method, scale, dataset_name)
                        if os.path.isdir(img_dir):
                            print(img_dir)
                            gts_ = os.listdir(gt_dir)
                            gts_ = sorted(gts_, key=lambda x: int(x[:-4]))
                            gts = [os.path.join(gt_dir, i) for i in gts_]
                            restoreds_ = os.listdir(img_dir)
                            restoreds_ = [i for i in restoreds_ if "jpg" in i or "png" in i]
                            restoreds_ = sorted(restoreds_, key=lambda x: int(x[:-4]))
                            restoreds = [os.path.join(img_dir, i) for i in restoreds_]

                            psnr_all, ssim_all = main3(gts, restoreds, True, crop_border=scale)
                            psnr_avg, psnr_std = np.mean(psnr_all), np.std(psnr_all)
                            ssim_avg, ssim_std = np.mean(ssim_all), np.std(ssim_all)
                        else:
                            psnr_avg, psnr_std, ssim_avg, ssim_std = "", "", "", ""
                        line.append(psnr_avg)
                        line.append(psnr_std)
                        line.append(ssim_avg)
                        line.append(ssim_std)
                    w.writerow(line)

    # 2. 与其他方法的对比(腹部数据集上的额外实验)
    if flag2:

        x4_flag = True
        x2_flag = False
        gt_dic = {
            47: "/mnt/c/data/测试数据集/3d-ircadb/hr_valid",
            174: "/mnt/c/data/belly_upload/hr_coronal",
            260: "/mnt/c/data/belly_upload/hr_sagittal"
        }
        gt_hu_dic = {
            -40: "/mnt/c/data/belly_upload/hr_multiHU/hu_-40_110",
            -100: "/mnt/c/data/belly_upload/hr_multiHU/hu_-100_900",
            -926: "/mnt/c/data/belly_upload/hr_multiHU/hu_-926_26"
        }
        if x4_flag:
            degration_names = ["hu_-40_110", "hu_-100_900", "hu_-926_26", "noise_artifact", "noise_gaussian", "noise_pepper", "down_cubic", "down_linear", "down_nearest", "loc_coronal", "loc_sagittal"][3:6]
            restored_root_dir = "/mnt/c/Users/NEU_s/OneDrive/document/想法/CT超分/补充实验/MultiDegrationExperiments/zt_{}_x4_extra_fornoise/visualization/{}"
            csv_file = f"/mnt/c/Users/NEU_s/OneDrive/document/想法/CT超分/补充实验/统计分析/multi_degration_x4_fornoise.csv"
            with open(csv_file, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                head = ["method"]
                for i in degration_names:
                    head.append(i)
                    head.append("")
                    head.append("")
                    head.append("")
                sub_head = [""] + ["psnr_avg", "psnr_std", "ssim_avg", "ssim_std"] * len(degration_names)
                w.writerow(head)
                w.writerow(sub_head)
                for method in methods:
                    line = [method]
                    for degration_name in degration_names:
                        img_dir = restored_root_dir.format(method, degration_name)
                        if os.path.isdir(img_dir):
                            print(img_dir)
                            if "hu" not in degration_name:
                                gt_dir = gt_dic[len(os.listdir(img_dir))]
                            else:
                                gt_dir = gt_hu_dic[int(degration_name.split("_")[1])]
                            gts_ = os.listdir(gt_dir)
                            gts_ = sorted(gts_, key=lambda x: int(x[:-4]))
                            gts = [os.path.join(gt_dir, i) for i in gts_]
                            restoreds_ = os.listdir(img_dir)
                            restoreds_ = [i for i in restoreds_ if "jpg" in i or "png" in i]
                            restoreds_ = sorted(restoreds_, key=lambda x: int(x[:-4]))
                            restoreds = [os.path.join(img_dir, i) for i in restoreds_]

                            psnr_all, ssim_all = main3(gts, restoreds, True, crop_border=4)
                            psnr_avg, psnr_std = np.mean(psnr_all), np.std(psnr_all)
                            ssim_avg, ssim_std = np.mean(ssim_all), np.std(ssim_all)
                        else:
                            print("dir is not exist: ", img_dir)
                            psnr_avg, psnr_std, ssim_avg, ssim_std = "", "", "", ""
                        line.append(psnr_avg)
                        line.append(psnr_std)
                        line.append(ssim_avg)
                        line.append(ssim_std)
                    w.writerow(line)



        if x2_flag:
            degration_names = ["hu_-40_110", "hu_-100_900", "hu_-926_26", "noise_artifact", "noise_gaussian",
                               "noise_pepper", "down_cubic", "down_linear", "down_nearest"][6:]
            restored_root_dir = "/mnt/c/Users/NEU_s/OneDrive/document/想法/CT超分/补充实验/MultiDegrationExperiments/zt_{}_x2_extra_fordown/visualization/{}"
            csv_file = f"/mnt/c/Users/NEU_s/OneDrive/document/想法/CT超分/补充实验/统计分析/multi_degration_x2_fordown_ignn.csv"
            with open(csv_file, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                head = ["method"]
                for i in degration_names:
                    head.append(i)
                    head.append("")
                    head.append("")
                    head.append("")
                sub_head = [""] + ["psnr_avg", "psnr_std", "ssim_avg", "ssim_std"] * len(degration_names)
                w.writerow(head)
                w.writerow(sub_head)
                for method in methods:
                    line = [method]
                    for degration_name in degration_names:
                        img_dir = restored_root_dir.format(method, degration_name)
                        if os.path.isdir(img_dir):
                            print(img_dir)
                            gt_dir = gt_dic[len(os.listdir(img_dir))]
                            gts_ = os.listdir(gt_dir)
                            gts_ = sorted(gts_, key=lambda x: int(x[:-4]))
                            gts = [os.path.join(gt_dir, i) for i in gts_]
                            restoreds_ = os.listdir(img_dir)
                            restoreds_ = [i for i in restoreds_ if "jpg" in i or "png" in i]
                            restoreds_ = sorted(restoreds_, key=lambda x: int(x[:-4]))
                            restoreds = [os.path.join(img_dir, i) for i in restoreds_]

                            psnr_all, ssim_all = main3(gts, restoreds, True, crop_border=2)
                            psnr_avg, psnr_std = np.mean(psnr_all), np.std(psnr_all)
                            ssim_avg, ssim_std = np.mean(ssim_all), np.std(ssim_all)
                        else:
                            psnr_avg, psnr_std, ssim_avg, ssim_std = "", "", "", ""
                        line.append(psnr_avg)
                        line.append(psnr_std)
                        line.append(ssim_avg)
                        line.append(ssim_std)
                    w.writerow(line)

    # 3. 消融实验
    if flag3:
        gt_dir = "/mnt/c/data/测试数据集/3d-ircadb/hr_valid"
        restored_root_dir = "/mnt/c/Users/NEU_s/OneDrive/document/想法/CT超分/补充实验/AblationExperiments/test_GHASR_x8_{}/visualization/3d-ircadb"
        modules = ["noscam", "nogffm"]
        csv_file = f"/mnt/c/Users/NEU_s/OneDrive/document/想法/CT超分/补充实验/统计分析/ablation.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            head = ["module", "3d-ircadb", "", "", ""]
            sub_head = ["", "psnr_avg", "psnr_std", "ssim_avg", "ssim_std"]
            w.writerow(head)
            w.writerow(sub_head)
            for module in modules:
                line = [module]
                img_dir = restored_root_dir.format(module)
                if os.path.isdir(img_dir):
                    print(img_dir)
                    gts_ = os.listdir(gt_dir)
                    gts_ = sorted(gts_, key=lambda x: int(x[:-4]))
                    gts = [os.path.join(gt_dir, i) for i in gts_]
                    restoreds_ = os.listdir(img_dir)
                    restoreds_ = [i for i in restoreds_ if "jpg" in i or "png" in i]
                    restoreds_ = sorted(restoreds_, key=lambda x: int(x[:-4]))
                    restoreds = [os.path.join(img_dir, i) for i in restoreds_]

                    psnr_all, ssim_all = main3(gts, restoreds, True, crop_border=4)
                    psnr_avg, psnr_std = np.mean(psnr_all), np.std(psnr_all)
                    ssim_avg, ssim_std = np.mean(ssim_all), np.std(ssim_all)
                else:
                    psnr_avg, psnr_std, ssim_avg, ssim_std = "", "", "", ""
                line.append(psnr_avg)
                line.append(psnr_std)
                line.append(ssim_avg)
                line.append(ssim_std)
                w.writerow(line)


if __name__ == '__main__':
    # restored_root = '/mnt/c/data/denoise_results/dwt_result'
    # restored_dir = [f"/mnt/c/data/hxy/results/proposed/x{i}/test_DANx2_l1loss" for i in [2, 4]]
    # txt_save_path = [f"/mnt/c/data/hxy/results/proposed/x{i}_l1loss.log" for i in [2, 4]]
    # for name in ["l1loss", "jointloss"]:
    #     restored_dir = [f"/mnt/c/data/hxy/results/proposed/x{i}/test_DANx{i}_{name}" for i in [2, 4]]
    #     txt_save_path = [f"/mnt/c/data/hxy/results/proposed/x{i}_{name}.log" for i in [2, 4]]
    #
    #     gt_dir = [f'/mnt/c/data/LUNA16/0/luna16dataset/nd/test'] * 2
    #     crop = [2, 4]
    #     for i in range(2):
    #         parser = argparse.ArgumentParser()
    #         parser.add_argument('--gt', type=str, default=gt_dir[i], help='Path to gt (Ground-Truth)')
    #         parser.add_argument('--restored', type=str, default=restored_dir[i],
    #                             help='Path to restored images')
    #         parser.add_argument('--crop_border', type=int, default=crop[i], help='Crop border for each side')
    #         parser.add_argument('--suffix', type=str, default='', help='Suffix for restored images')
    #         # parser.add_argument(
    #         #     '--test_y_channel',
    #         #     action='store_true',
    #         #     help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
    #         # parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    #         args = parser.parse_args()
    #         args.test_y_channel = False
    #         args.correct_mean_var = False
    #
    #         main(args, save_txt_path=txt_save_path[i], flag=cv2.IMREAD_GRAYSCALE)

    for data in ["3dircadb", "pancreas"]:
        for s in [2, 4]:
            for p in ["val", "test"]:
                gts = glob.glob(f"/home/zhiyi/data/{data}/img/hr_nd/{p}/*")
                restoreds = glob.glob(f"/home/zhiyi/data/{data}/img/lr_ld_bicubic/x{s}/{p}/*")
                print(f"{data} x{s} {p}")
                main3(gts, restoreds, test_y_channel=True, crop_border=s)
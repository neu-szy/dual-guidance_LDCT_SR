# 生成bicubic下采样的图片
import os
from PIL import Image
import numpy as np
import tqdm


def main(data_dir="/home/zhiyi/data/medical/belly", hr_name="hr", lr_name="x2", scale=2, lr_name_template=None):
    # Scale factor
    hr_dir = os.path.join(data_dir, hr_name)
    assert os.path.isdir(hr_dir), "hr路径不存在"
    lr_dir = os.path.join(data_dir, lr_name)
    if not os.path.isdir(lr_dir):
        os.makedirs(lr_dir)
    hr_list = os.listdir(hr_dir)

    for i in tqdm.tqdm(range(len(hr_list))):
        # print(f"第{i}张")
        img_hr = os.path.join(hr_dir, hr_list[i])
        img_hr = Image.open(img_hr)
        dsize = (img_hr.size[0] // scale, img_hr.size[1] // scale)
        img_hr = np.array(img_hr)
        if len(img_hr.shape) == 3:
            img_hr = img_hr[:img_hr.shape[0]//scale*scale, :img_hr.shape[1]//scale*scale, :]
        else:
            img_hr = img_hr[:img_hr.shape[0] // scale * scale, :img_hr.shape[1] // scale * scale]
        img_hr = Image.fromarray(img_hr)
        img_lr = img_hr.resize(dsize, Image.BICUBIC)
        if lr_name_template is None:
            lr_path = os.path.join(lr_dir, hr_list[i])
        else:
            b, e = os.path.splitext(hr_list[i])
            b = b.split("_")
            t = []
            for i in b:
                if i.isdigit():
                    t.append(i)
            lr_name = lr_name_template.format(*t)
            lr_path = os.path.join(lr_dir, lr_name)
        img_lr.save(lr_path)


if __name__ == "__main__":
    # d = ["B100", "manga109", "urban100"]
    # for i in [8]:
    #     main(f"/home/zhiyi/data/Set14", i)
    # for s in [2, 3, 4, 8]:
    #     main(r"/data/zy/DIV2K/val", s)

    # for s in [2, 4, 8]:
    #     main("/home/zhiyi/data/medical/belly", "hr_mask", f"x{s}_mask", s, "image_{}_{}.jpg")
    #     main("/home/zhiyi/data/medical/belly", "hr_mask_valid", f"x{s}_mask_valid", s, "image_{}_{}.jpg")

    # img_names = os.listdir("/home/zhiyi/data/medical/belly/hr_mask_origin")
    # save_dir = "/home/zhiyi/data/medical/belly/hr_mask"
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # for img_name in img_names:
    #     b, e = os.path.splitext(img_name)
    #     b = b.split("_")
    #     t = []
    #     for i in b:
    #         if i.isdigit():
    #             t.append(i)
    #     img = Image.open("/home/zhiyi/data/medical/belly/hr_mask_origin/" + img_name)
    #     save_path = os.path.join(save_dir, "image_{}_{}.jpg".format(*t))
    #     img.save(save_path)
    #
    # img_names = os.listdir("/home/zhiyi/data/medical/belly/hr_mask_valid_origin")
    # save_dir = "/home/zhiyi/data/medical/belly/hr_mask_valid"
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # for img_name in img_names:
    #     b, e = os.path.splitext(img_name)
    #     b = b.split("_")
    #     t = []
    #     for i in b:
    #         if i.isdigit():
    #             t.append(i)
    #     img = Image.open("/home/zhiyi/data/medical/belly/hr_mask_valid_origin/" + img_name)
    #     save_path = os.path.join(save_dir, "image_{}_{}.jpg".format(*t))
    #     img.save(save_path)

    for s in [2, 4]:
        for n in ["train", "val", "test"]:
            # main("/home/zhiyi/data/3dircadb/img", f"hr_ld/{n}", f"lr_ld/x{s}/{n}", s)
            main("/home/zhiyi/data/pancreas/mask_old", f"hr/{n}", f"x{s}/{n}", s)
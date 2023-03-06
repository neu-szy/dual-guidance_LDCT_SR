import os
from tqdm import tqdm
from PIL import Image

def main(dir, save_dir, scale):
    img_list = os.listdir(dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for img_name in tqdm(img_list):
        img_path = os.path.join(dir, img_name)
        img = Image.open(img_path)
        w, h = img.size
        img = img.resize((w*scale, h*scale), Image.BICUBIC)
        save_path = os.path.join(save_dir, img_name)
        img.save(save_path)

# for name in ["belly", "lung"]:
    # for scale in [2, 4, 8]:
    #     main(f"/home/zhiyi/data/medical/{name}/x{scale}_valid",
    #          f"/home/zhiyi/data/medical/{name}/x{scale}_valid_bicubic_sr",
    #          scale)

# for a, b in zip(
#         ["x4_coronal", "x4_sagittal", "x4_val_artifact1e3", "x4_val_cubic", "x4_val_linear", "x4_val_nearest", "x4_val_pepper", "x4_val_gaussian", "x4_multiHU/hu_-40_110", "x4_multiHU/hu_-100_900", "x4_multiHU/hu_-926_26"][3:6],
#         ["x4_coronal_bicubic_sr", "x4_sagittal_bicubic_sr", "x4_val_artifact1e3_bicubic_sr", "x4_val_cubic_bicubic_sr", "x4_val_linear_bicubic_sr", "x4_val_nearest_bicubic_sr", "x4_val_pepper_bicubic_sr", "x4_val_gaussian_bicubic_sr", "x4_multiHU/hu_-40_110_bicubic_sr", "x4_multiHU/hu_-100_900_bicubic_sr", "x4_multiHU/hu_-926_26_bicubic_sr"][3:6]
# ):
#     main(f"/home/zhiyi/data/medical/belly/zt_extra/{a}", f"/home/zhiyi/data/medical/belly/zt_extra/{b}", 4)

# for a, b in zip(
#     ["x2_val_cubic", "x2_val_linear", "x2_val_nearest"],
#     ["x2_val_cubic_bicubic_sr", "x2_val_linear_bicubic_sr", "x2_val_nearest_bicubic_sr"]
# ):
#     main(f"/home/zhiyi/data/medical/belly/zt_extra/{a}", f"/home/zhiyi/data/medical/belly/zt_extra/{b}", 2)


if __name__ == '__main__':
    for data in ["3dircadb", "pancreas"]:
        for s in [2, 4]:
            for p in ["val", "test"]:
                main(f"/home/zhiyi/data/{data}/img/lr_ld/x{s}/{p}", f"/home/zhiyi/data/{data}/img/lr_ld_bicubic/x{s}/{p}", s)
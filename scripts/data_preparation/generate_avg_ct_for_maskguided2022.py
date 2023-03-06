import cv2
import os
import glob

import numpy as np


def main(lq_dir, save_dir, scale):
    os.makedirs(save_dir, exist_ok=True)
    img_names = os.listdir(lq_dir)
    num1s = [os.path.splitext(i)[0].split("_")[0] for i in img_names]
    num1s = list(set(num1s))
    for num1 in num1s:
        img_total = np.zeros((512 // scale, 512 // scale), dtype=np.float64)
        n = 0
        img_num1_paths = glob.glob(os.path.join(lq_dir, f"{num1}_*.*"))
        for img_num1_path in img_num1_paths:
            img = cv2.imread(img_num1_path, flags=cv2.IMREAD_GRAYSCALE)
            img_total += img
            n += 1
        img_avg = img_total / n
        cv2.imwrite(os.path.join(save_dir, f"{num1}.png"), np.uint8(img_avg))

if __name__ == '__main__':
    for d in ["3dircadb", "pancreas"]:
        for s in [2, 4]:
            for p in ["train", "val", "test"]:
                main(f"/home/zhiyi/data/{d}/img/lr_nd/x{s}/{p}", f"/home/zhiyi/data/{d}/img/lr_nd/x{s}/{p}_avg", s)
# README
The official implement of our paper which was submitted to MICCAI 2023.

This repository is modified from [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks for the open source code of [BasicSR](https://github.com/XPixelGroup/BasicSR).
## Installation
```bash
conda create -n new_env python=3.9.7 -y
conda activate new_env
pip install -r requirements.txt
pip install -e .
```
More details could be found in [the installation ducoment of Basicsr](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md).
## Data preparation
You should prepare your data in this way:
```
data_rootdir
    - dataset_name
        - img
            - hr_nd
                - train
                - val
                - test
            - lr_ld
                - x2
                    - train
                    - train_avg
                    - val
                    - val_avg
                    - test
                    - test_avg
                - x4
                    - train
                    - train_avg
                    - val
                    - val_avg
                    - test
                    - test_avg
            - lr_nd
                - x2
                    - train
                    - val
                    - test
                - x4
                    - train
                    - val
                    - test
        -mask
            - hr
                - train
                - val
                - test
            - x2
                - train
                - val
                - test
            - x4
                - train
                - val
                - test
```
And you should modify the path in configuration files in "opations/train/*.yaml" or "opations/test/*.yaml".
## Training
Run:
```bash
python basicsr/train.py --opt options/train/your_config_file.yml
```
The model files will be saved in "experiments" folder.
## Testing
Firstly, you should modify the model paths in "opations/test/*.yaml".
Then, run:
```bash
python basicsr/test.py --opt options/test/your_config_file.yml
```
The results will be saved in "results" folder.

An example, including models and dataset, could be found in [BaiduDisk:z3gy](https://pan.baidu.com/s/1l7lXLCOJWeQVZOt_ldtGbg).
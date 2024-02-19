# copy from run.py
# to find a best th for inference

import glob
import os

from model import SARModel
from dataset import SARdataset
import albumentations as A

import lightning as L
import os
import glob
import torch
from model import SARModel
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import cv2
from dataset import test_trans
import ttach as tta
from tool import read_img2tensor
import torchmetrics as tm
from tool import f1_func
import pickle

# ------- param ------
th = 0.5
use_tta = False

# ------- PATH ---------
project_p = "/home/ao/Desktop/ieee/"
train_data_p = project_p + "data/Track1/train/images/"
train_label_p = project_p + "data/Track1/train/labels/"
# test_data_p = project_p + "data/Track1/val/images/"
# dev_data_p = project_p + "data/dev/p1"

# ---- infer -----

batch_size = 4
# ckpt
model_p_list = [
    # 0.9220
    # "/home/ao/Desktop/ieee/rubbish/sar/c5aht82z/checkpoints/epoch=287-step=105696.ckpt",
    # 0.9240
    "/home/ao/Desktop/ieee/ckpt/epoch=227-step=83676.ckpt",
]

# tta
tta_com = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
        # tta.Scale(scales=[1, 2, 4]),
        # tta.Multiply(factors=[0.9, 1, 1.1]),
    ]
)

if __name__ == "__main__":
    os.system("rm -f /home/ao/Desktop/ieee/rubbish/sub.zip")
    os.system("rm -f /home/ao/Desktop/ieee/rubbish/sub/* ")

    # img_l = glob.glob(os.path.join(train_data_p, "*.tif"))
    img_l = glob.glob(os.path.join(train_data_p, "*.tif"))
    label_l = glob.glob(os.path.join(train_label_p, "*.png"))
    label_l.sort()
    img_l.sort()

    # for test
    # img_l=img_l[:10]
    # label_l=label_l[:10]

    dataset = SARdataset(img_l, normal=True)
    dataset.transform = test_trans

    test_loader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count() - 1
    )
    # test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=os.cpu_count()-1)

    th_l = [0.5, 0.49, 0.495, 0.505, 0.46]

    # th test
    for i in th_l:
        th = i

        # ---- run -------
        f1_score_l = []
        f1_score_dict = {}

        with torch.inference_mode():
            for p in tqdm(test_loader):
                batch_ans = []
                for path in tqdm(model_p_list, leave=False):
                    # ---- model set ----
                    sar_model = SARModel.load_from_checkpoint(
                        path,
                        arch="UnetPlusPlus",
                        encoder_name="timm-resnest269e",
                        in_channels=6,
                        encoder_weights="imagenet",
                    )
                    sar_model.eval()

                    # ---- tta_warpper ----
                    if use_tta:
                        tta_model = tta.SegmentationTTAWrapper(sar_model.model, tta_com)
                    else:
                        tta_model = sar_model.model

                    if torch.cuda.is_available():
                        tta_model.to("cuda")

                    # ---- infer ----
                    if torch.cuda.is_available():
                        p[0] = p[0].to("cuda")

                    # get one model ans and fetch to cpu
                    batch_ans.append(tta_model(p[0]).to("cpu"))

                ans = sum(batch_ans) / len(batch_ans)
                # ans = torch.mean(torch.stack(batch_ans), dim=0)

                # p[1]
                pic = (ans.sigmoid() > th).float()

                for i_pic, index in zip(pic, p[1]):
                    # i_pic -> (1, 512, 512)
                    pred = i_pic[0].to("cpu")
                    mask = read_img2tensor(label_l[index])
                    f1_score = f1_func(pred.flatten(), mask.flatten())
                    f1_score_l.append(f1_score.item())
                    f1_score_dict[img_l[index.item()]] = f1_score.item()
                    f1_score_dict_sorted = dict(
                        sorted(f1_score_dict.items(), key=lambda x: x[1])
                    )

        f = open(f"th-{th}-mean-{np.mean(f1_score_l)}.pkl", "wb")
        pickle.dump(f1_score_dict_sorted, f)
        print(np.mean(f1_score_l))

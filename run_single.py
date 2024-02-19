import glob
import os

from model import SARModel
from dataset import SARdataset
import albumentations as A

import lightning as L
import os
import glob

from model import SARModel
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import cv2
from dataset import test_trans
import ttach as tta
import torch

# ------- 变量 ---------
project_p = "/home/ao/Desktop/ieee/"
train_data_p = project_p + "data/Track1/train/images/"
train_label_p = project_p + "data/Track1/train/labels/"
test_data_p = project_p + "data/Track1/val/images/"
dev_data_p = project_p + "data/dev/p1"

# ---- infer -----
th = 0.0001
# th = 0.001 -> 0.918
# th = 0.01
# th = 0.03 -> 0.926

batch_size = 1
# ckpt
# model_p = "/home/ao/Desktop/ieee/rubbish/sar/xz50i3nu/checkpoints/val_score_epoch=102-val_loss=0.0913-score_f1_val=0.9045.ckpt"
model_p = (
    "/home/ao/Desktop/ieee/rubbish/sar/lt2zc1hj/checkpoints/epoch=227-step=83676.ckpt"
)

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

    # img_l = glob.glob(os.path.join(train_data_p, '*.tif'))
    # label_l = glob.glob(os.path.join(train_label_p, '*.png'))
    # label_l.sort()

    # ---- dataset ----
    img_l = glob.glob(os.path.join(dev_data_p, "*.tif"))
    img_l.sort()

    dataset = SARdataset(img_l, normal=True)
    dataset.transform = test_trans
    test_loader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count() - 1
    )

    sar_model = SARModel.load_from_checkpoint(
        model_p,
        arch="UnetPlusPlus",
        encoder_name="timm-resnest269e",
        # encoder_name="resnet34",
        in_channels=6,
        encoder_weights="imagenet",
    )
    sar_model.eval()

    with torch.inference_mode():
        # ---- tta_warpper ----
        tta_model = tta.SegmentationTTAWrapper(sar_model.model, tta_com)
        if torch.cuda.is_available():
            tta_model.to("cuda")

        # ---- infer ----
        with torch.inference_mode():
            for p in tqdm(test_loader):
                if torch.cuda.is_available():
                    p[0] = p[0].to("cuda")
                ans = tta_model(p[0])
                pic = (ans.sigmoid() > th).float()
                for i_pic, index in zip(pic, p[1]):
                    # i_pic -> (1, 512, 512)
                    data = i_pic[0].to("cpu")
                    image = data.numpy().astype(np.uint8)
                    cv2.imwrite("sub/" + f"{index+1631}_msk.png", image)

    os.system("cd sub/ &&  zip -rv ../sub.zip *.png ")

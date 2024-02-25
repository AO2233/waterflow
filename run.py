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

# ------- 变量 ---------
project_p = "/home/ao/Desktop/ieee/"
train_data_p = project_p + "data/Track1/train/images/"
train_label_p = project_p + "data/Track1/train/labels/"
test_data_p = project_p + "data/Track1/val/images/"
dev_data_p = project_p + "data/dev/p1"

# ---- infer -----
th = 0.1
batch_size = 16
# ckpt
model_p_list = [
    "/home/ao/Desktop/ieee/rubbish/ema-new-KFP/665fzo1b/checkpoints/val_score_fold0_epoch=262-val_loss=0.0921-score_f1_val=0.9398.ckpt",
    # '/home/ao/Desktop/ieee/rubbish/ema-new-KFP/7ohddt21/checkpoints/val_score_fold2_epoch=162-val_loss=0.1199-score_f1_val=0.9091.ckpt',
    # '/home/ao/Desktop/ieee/rubbish/ema-new-KFP/g58gxa3u/checkpoints/val_score_fold4_epoch=233-val_loss=0.0808-score_f1_val=0.9301.ckpt',
    # '/home/ao/Desktop/ieee/rubbish/ema-new-KFP/pimbrg8x/checkpoints/val_score_fold3_epoch=283-val_loss=0.1028-score_f1_val=0.9105.ckpt',
    "/home/ao/Desktop/ieee/rubbish/ema-new-KFP/zgiqcalm/checkpoints/val_score_fold1_epoch=218-val_loss=0.0944-score_f1_val=0.9418.ckpt",
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

    # img_l = glob.glob(os.path.join(train_data_p, '*.tif'))
    # label_l = glob.glob(os.path.join(train_label_p, '*.png'))
    # label_l.sort()

    img_l = glob.glob(os.path.join(dev_data_p, "*.tif"))
    img_l.sort()

    dataset = SARdataset(img_l, normal=True)
    dataset.transform = test_trans

    test_loader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=os.cpu_count() - 1
    )
    # test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=os.cpu_count()-1)

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
                tta_model = tta.SegmentationTTAWrapper(sar_model.model, tta_com)
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
                data = i_pic[0].to("cpu")
                image = data.numpy().astype(np.uint8)
                cv2.imwrite("sub/" + f"{index+1631}_msk.png", image)

            # for p in tqdm(test_loader):
            #     # p[0] = p[0].to('cuda')
            #     ans = sar_model(p)
            #     pic = (ans[0].sigmoid() > th).float()
            #     for i_pic, index in zip(pic, ans[1]):
            #         data = i_pic[0].to('cpu')
            #         image =data.numpy().astype(np.uint8)
            #         cv2.imwrite("sub/"+f"{index+1631}_msk.png", image)

    # zip and upload
    os.system("cd sub/ &&  zip -rv ../sub.zip *.png ")

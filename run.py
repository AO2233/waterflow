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
project_p="/home/ao/Desktop/ieee/"
train_data_p = project_p + "data/Track1/train/images/"
train_label_p = project_p + "data/Track1/train/labels/"
test_data_p = project_p + "data/Track1/val/images/"
dev_data_p = project_p + "data/dev/p1"

# ---- infer -----
th = 0.5
batch_size = 1
# ckpt
model_p = '/home/ao/Desktop/ieee/rubbish/epoch=170-step=62757.ckpt'

if __name__ == "__main__":
    
    tta_com = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
            # tta.Scale(scales=[1, 2, 4]),
            # tta.Multiply(factors=[0.9, 1, 1.1]),        
            ]
        )
    
    os.system("rm -f /home/ao/Desktop/ieee/rubbish/sub.zip")
    os.system("rm -f /home/ao/Desktop/ieee/rubbish/sub/* ")
    
    # img_l = glob.glob(os.path.join(train_data_p, '*.tif'))
    # label_l = glob.glob(os.path.join(train_label_p, '*.png'))
    # label_l.sort()
    
    img_l = glob.glob(os.path.join(dev_data_p, '*.tif'))
    img_l.sort()
    
    sar_model=SARModel.load_from_checkpoint(
        model_p, 
        arch="UnetPlusPlus",
        encoder_name="timm-resnest269e",
        # encoder_name="resnet34",
        in_channels=6, 
        encoder_weights = 'imagenet'
        )
    
    sar_model.eval()
    
    tta_model = tta.SegmentationTTAWrapper(sar_model.model, tta_com)
    tta_model.to('cuda')
    
    dataset = SARdataset(img_l, normal=True)
    dataset.transform = test_trans
    
    test_loader = DataLoader(dataset, batch_size=batch_size)
    # test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=os.cpu_count()-1)

    with torch.inference_mode():
        for p in tqdm(test_loader):
            p[0] = p[0].to('cuda')
            ans = [tta_model(p[0]), p[1]]
            pic = (ans[0].sigmoid() > th).float()
            for i_pic, index in zip(pic, ans[1]):
                data = i_pic[0].to('cpu')
                image =data.numpy().astype(np.uint8)
                cv2.imwrite("sub/"+f"{index+1631}_msk.png", image)
    
    
        # for p in tqdm(test_loader):
        #     # p[0] = p[0].to('cuda')
        #     ans = sar_model(p)
        #     pic = (ans[0].sigmoid() > th).float()
        #     for i_pic, index in zip(pic, ans[1]):
        #         data = i_pic[0].to('cpu')
        #         image =data.numpy().astype(np.uint8)
        #         cv2.imwrite("sub/"+f"{index+1631}_msk.png", image)
    
    os.system("cd sub/ &&  zip -rv ../sub.zip *.png ")    
    

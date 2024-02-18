import numpy as np
import cv2
import tifffile as tf
import matplotlib.pyplot as plt
import albumentations as A
import torch
import glob
import os
import torchmetrics as tm


def show_tiff(image, label=None, power=True):
    plt.figure(figsize=(12, 6))

    assert len(image.shape) == 3 and image.shape[2]
    for i in range(6):
        plt.subplot(2, 4, i + 1)
        plt.imshow(image[:, :, i], cmap="gray")
        plt.title(f"Channel {i+1}")
        plt.axis("off")
    if label is not None:
        if power:
            label = 255 * (1 - label)
        plt.subplot(2, 4, 7)
        plt.imshow(label, cmap="gray")
        plt.title(f"label")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def show_tiff_id(idx, img_l, label_l=None, power=True):
    image = tf.imread(img_l[idx])
    image_label = None
    if label_l is not None:
        image_label = cv2.imread(label_l[idx])
    show_tiff(image, label=image_label, power=power)

def show_label(label, power=True):
    if power:
        label = 255 * (1 - label)
    plt.figure(figsize=(12, 6))
    plt.imshow(label, cmap="gray")
    plt.title(f"label")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def check_sub(dir="sub", power=True):
    sub_pic = glob.glob(os.path.join(dir, "*.png"))
    sub_pic.sort()

    assert len(sub_pic) >= 12

    plt.figure(figsize=(16, 8))
    for i in range(12):
        label = cv2.imread(sub_pic[i], cv2.IMREAD_UNCHANGED)
        if power:
            label = 255 * (1 - label)

        plt.subplot(3, 4, i + 1)
        plt.imshow(label, cmap="gray")
        plt.title(f"item {i+1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def read_img2tensor(path):
    label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return torch.tensor(label, dtype=torch.float32)


# 用于修正 原版本的 F1Score zero-dive 等于0的问题
if torch.cuda.is_available():
    warp_base_func = tm.F1Score(task="binary").to('cuda')
else:
    warp_base_func = tm.F1Score(task="binary")
    
def f1_func(pred,mask): 
    if torch.all(mask == 0) and torch.all(pred == 0):
        f1_score = torch.tensor(1.0)
    else:
        f1_score = warp_base_func(pred.flatten(), mask.flatten())
    
    return f1_score
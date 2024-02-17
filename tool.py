import numpy as np
import cv2
import tifffile as tf
import matplotlib.pyplot as plt
import albumentations as A
import torch


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

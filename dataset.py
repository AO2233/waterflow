import torch
import numpy as np
import cv2
import tifffile as tf
from torch.utils.data import Dataset, random_split
import albumentations as A
import albumentations.pytorch.transforms as T
import pickle
import copy


class SARdataset(Dataset):
    # val mode mask = None
    def __init__(
        self, sar_list, mask_list=None, mode="test", transform=None, normal=False
    ):
        if mode == "test":
            assert mask_list is None

        self.sar_list = sar_list
        self.mask_list = mask_list
        self.transform = transform
        self.normal = normal

    def __len__(self):
        return len(self.sar_list)

    def __getitem__(self, idx):
        assert self.transform is not None

        sar = tf.imread(self.sar_list[idx])
        sar = sar.astype(np.float32)

        if self.mask_list != None:
            mask = cv2.imread(self.mask_list[idx], cv2.IMREAD_UNCHANGED)

            # 512 x 512 x 1
            mask = np.expand_dims(mask, axis=-1)
            mask = mask.astype(np.float32)
        else:
            mask = None

        if mask is not None:
            transformed = self.transform(image=sar, mask=mask)
            sar = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = self.transform(image=sar)
            sar = transformed["image"]

        if self.normal:
            min_l = [0.0, 0.0, -9999.0, -33.0, 0.0, 0.0]
            max_l = [32765.0, 32765.0, 3343.0, 3310.0, 100.0, 255.0]
            std_l = [
                2797.2916273636556,
                710.5693234469384,
                1173.4579685284687,
                342.08553456322215,
                17.937904473111903,
                17.983585131110953,
            ]
            mean_l = [
                1774.172843321052,
                382.70600456948034,
                32.83576673379225,
                158.46333796412378,
                31.623006319089388,
                3.7178739697469863,
            ]

            # sar[0] = (sar[0]-mean_l[0]) / std_l[0]

            for i in range(len(sar)):
                sar[i] = (sar[i] - mean_l[i]) / std_l[i]

        if mask is not None:
            return sar, mask, idx  # idx 仅在infer的时候用到
        else:
            return sar, idx


train_trans = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf(
            [
                A.Rotate(limit=(90, 90)),
                A.Rotate(limit=(-90, -90)),
                A.Rotate(limit=(180, 180)),
                A.Rotate(limit=(-180, -180)),
            ],
            p=0.5,
        ),
        # A.RandomResizedCrop(width=512,height=512, p=0.3)
        # A.OneOf([A.RandomResizedCrop(width=512,height=512),A.RandomGridShuffle()],p=0.3),
        # A.OneOf([A.GridDistortion(), A.OpticalDistortion()], p=0.3),
        # A.OneOf([
        #             A.GaussianBlur(),
        #             A.GaussNoise(var_limit=(0, 2e-8), mean=0, per_channel=True),
        #             ],
        #             p=0.2),
        T.ToTensorV2(transpose_mask=True),
    ]
)

test_trans = A.Compose(
    [
        T.ToTensorV2(transpose_mask=True),
    ]
)

unchange_trans = A.Compose(
    [
        T.ToTensorV2(transpose_mask=True),
    ]
)


def build_dataset(dataset, val_rate):
    t_dataset = copy.deepcopy(dataset)
    val_size = int(val_rate * len(dataset))
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(t_dataset, [train_size, val_size])
    val_set.dataset = copy.deepcopy(t_dataset)

    train_set.dataset.transform = train_trans
    val_set.dataset.transform = test_trans

    return train_set, val_set


if __name__ == "__main__":
    from dataset import *
    import glob
    import os

    project_p = "/home/ao/Desktop/ieee/"
    train_data_p = project_p + "data/Track1/train/images/"
    train_label_p = project_p + "data/Track1/train/labels/"
    test_data_p = project_p + "data/Track1/val/images/"

    img_l = glob.glob(os.path.join(train_data_p, "*.tif"))
    label_l = glob.glob(os.path.join(train_label_p, "*.png"))

    img_l.sort()
    label_l.sort()

    dataset = SARdataset(img_l)
    train_set, val_set = build_dataset(dataset, 0.2)

    print(len(train_set), len(val_set) / 16)

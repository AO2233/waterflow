import lightning as L
import os
import glob

from dataset import *
from model import SARModel
from torch.utils.data import DataLoader
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data.dataset import Subset
from sklearn import model_selection


from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)


# -------- 设置 ---------
project_p = "/home/ao/Desktop/ieee/"
train_data_p = project_p + "data/Track1/train/images/"
train_label_p = project_p + "data/Track1/train/labels/"
test_data_p = project_p + "data/Track1/val/images/"


# --------- 参数 ---------
max_ep = 360
val_rate = 0.1
batch_size = 4

# img_num = 1631

## callback pylighting model setting

SEED = 42
seed_everything(SEED, workers=True)
lr_monitor = LearningRateMonitor("epoch")
progress_bar = RichProgressBar()
model_summary = RichModelSummary(max_depth=3)

# ---------------------------

if __name__ == "__main__":
    img_l = glob.glob(os.path.join(train_data_p, "*.tif"))
    label_l = glob.glob(os.path.join(train_label_p, "*.png"))

    img_l.sort()
    label_l.sort()

    # img_l = [train_data_p + f"{i}.tif" for i in range(img_num)]
    # label_l = [project_p + f"data/Track1/train/labels/{i}.png" for i in range(img_num)]

    tt_dataset = SARdataset(img_l, label_l, mode="train", normal=True)
    tv_dataset = copy.deepcopy(tt_dataset)
    tt_dataset.transform = train_trans
    tv_dataset.transform = test_trans

    kf_func = model_selection.KFold(
        n_splits=int(1 / val_rate), random_state=SEED, shuffle=True
    )
    # kf_func = model_selection.KFold(n_splits = int(1/val_rate), shuffle=False)

    # [train_index: List, val_index:List] * K_Fold_num
    kf_l = kf_func.split(tt_dataset)

    for i, (train_index, val_index) in enumerate(kf_l):
        train_dataset = Subset(tt_dataset, train_index)
        val_dataset = Subset(tv_dataset, val_index)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count() - 1,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count() - 1,
            drop_last=True,
        )
        model = SARModel(
            "UnetPlusPlus",
            encoder_name="timm-resnest269e",
            in_channels=6,
            encoder_weights="imagenet",
        )
        # model=SARModel("UnetPlusPlus", encoder_name="resnet34", in_channels=6, encoder_weights = 'imagenet')

        # ------ wandb --------
        wandb_logger = WandbLogger(
            project="KFP",
            # log_model="all",
            name=f"KF{i}",
        )

        loss_checkpoint_callback = ModelCheckpoint(
            verbose=True,
            filename=f"val_loss_fold{i}_" + "{epoch}-{val_loss:.4f}-{score_f1_val:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=20,
            save_last=True,
            save_weights_only=True,
            auto_insert_metric_name=True,
            # every_n_epochs = max_ep//20,
        )

        score_checkpoint_callback = ModelCheckpoint(
            verbose=True,
            filename=f"val_score_fold{i}_"
            + "{epoch}-{val_loss:.4f}-{score_f1_val:.4f}",
            monitor="score_f1_val",
            save_top_k=5,
            save_weights_only=True,
            mode="max",
            auto_insert_metric_name=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        # ---------------
        trainer = L.Trainer(
            accelerator="auto",
            logger=wandb_logger,
            max_steps=max_ep * len(train_loader),
            max_epochs=max_ep,
            gradient_clip_val=1.0,
            # accumulate_grad_batches = 1,
            # sync_batchnorm=True,
            callbacks=[
                loss_checkpoint_callback,
                score_checkpoint_callback,
                lr_monitor,
                progress_bar,
                model_summary,
            ],
        )

        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        wandb.finish()
        # print(trainer.max_steps)

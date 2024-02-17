import lightning as L
import os
import glob

from dataset import *
from model import SARModel
from torch.utils.data import DataLoader
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)


# -------- 设置 ---------
project_p="/home/ao/Desktop/ieee/"
train_data_p = project_p + "data/Track1/train/images/"
train_label_p = project_p + "data/Track1/train/labels/"
test_data_p = project_p + "data/Track1/val/images/"


# --------- 参数 ---------
max_ep = 400
val_rate = 0.1
batch_size = 4

# img_num = 1631

## callback pylighting model setting 
seed_everything(42, workers=True)
lr_monitor = LearningRateMonitor("epoch")
progress_bar = RichProgressBar()
model_summary = RichModelSummary(max_depth=2)

# ---------------------------

if __name__ == "__main__":
    img_l = glob.glob(os.path.join(train_data_p, '*.tif'))
    label_l = glob.glob(os.path.join(train_label_p, '*.png'))
    
    img_l.sort()
    label_l.sort()
    
    # img_l = [train_data_p + f"{i}.tif" for i in range(img_num)]
    # label_l = [project_p + f"data/Track1/train/labels/{i}.png" for i in range(img_num)]
    dataset = SARdataset(img_l, label_l, mode='train',normal=True)
    
    train_set, val_set = build_dataset(dataset, val_rate)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=os.cpu_count()-1, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size*2, shuffle=False, pin_memory=True, num_workers=os.cpu_count()-1, drop_last=True)
    model=SARModel("UnetPlusPlus", encoder_name="timm-resnest269e", in_channels=6, encoder_weights = 'imagenet')
    # model=SARModel("UnetPlusPlus", encoder_name="resnet34", in_channels=6, encoder_weights = 'imagenet')
    
    # ------ wandb --------
    wandb_logger = WandbLogger(
        project="sar",
        # log_model="all",
        name="test",
        )
    
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor="val_loss", 
        mode="min",
        save_top_k = 20,
        save_last = True,
        save_weights_only = True,
        # every_n_epochs = max_ep//20,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # ---------------
    
    trainer = L.Trainer(
        accelerator="auto",
        logger=wandb_logger,
        max_steps=max_ep*len(train_loader),
        max_epochs=max_ep,
        gradient_clip_val = 1.0,
        # accumulate_grad_batches = 1,
        # sync_batchnorm=True,
        callbacks=[checkpoint_callback, lr_monitor, progress_bar, model_summary]
        )
    
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
        )
    
    # print(trainer.max_steps)
from typing import Any
import numpy as np
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torchmetrics as tm
import torch.optim as optim
import segmentation_models_pytorch as smp
import timm
from transformers import Swinv2Config, Swinv2Model
from torch import nn
import torch
from transformers import get_cosine_schedule_with_warmup
from ema import EMAOptimizer


def loss_warp(output, mask):
    dice_r = 0.75
    loss_Dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    loss_BCE = nn.BCEWithLogitsLoss()
    loss = (1 - dice_r) * loss_BCE(output, mask) + dice_r * loss_Dice(output, mask)
    return loss


class SARModel(L.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, encoder_weights=None):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
        # self.model = timm.create_model('convnextv2_huge.fcmae_ft_in22k_in1k_512', pretrained=True)

        # self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = loss_warp

        self.f1_func = tm.F1Score(task="binary")
        if torch.cuda.is_available():
            self.f1_func = tm.F1Score(task="binary")

        self.validation_step_outputs_mask = []
        self.validation_step_outputs_pre = []
        self.train_step_outputs_mask = []
        self.train_step_outputs_pre = []

    def forward(self, batch):
        output = self.model(batch[0])
        return output

    def training_step(self, batch, batch_idx):
        # sar, mask, idx
        image = batch[0]
        mask = batch[1]

        # img -> (bs, 6, h, w)
        # output & mask (bs, 1, h, w)
        output = self.model(image)
        train_loss = self.loss_fn(output, mask)

        sig_mask = output.sigmoid() > 0.5
        pred_mask = (sig_mask).float()

        self.train_step_outputs_mask.append(mask.flatten().to("cpu"))
        self.train_step_outputs_pre.append(pred_mask.flatten().to("cpu"))

        self.log(
            "train_loss",
            train_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        image = batch[0]
        mask = batch[1]
        output = self.model(image)
        val_loss = self.loss_fn(output, mask)

        sig_mask = output.sigmoid() > 0.5
        pred_mask = (sig_mask).float()

        self.validation_step_outputs_mask.append(mask.flatten().to("cpu"))
        self.validation_step_outputs_pre.append(pred_mask.flatten().to("cpu"))

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return val_loss

    def on_train_epoch_end(self):
        all_mask = torch.cat(self.train_step_outputs_mask)
        all_pre = torch.cat(self.train_step_outputs_pre)

        f1_fin_train = self.f1_func(all_pre, all_mask)

        self.train_step_outputs_mask.clear()
        self.train_step_outputs_pre.clear()

        self.log(
            "score_f1_train",
            f1_fin_train,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self):
        all_mask = torch.cat(self.validation_step_outputs_mask)
        all_pre = torch.cat(self.validation_step_outputs_pre)

        f1_fin_val = self.f1_func(all_pre, all_mask)

        self.validation_step_outputs_mask.clear()
        self.validation_step_outputs_pre.clear()

        self.log(
            "score_f1_val",
            f1_fin_val,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=4e-5, weight_decay=4e-3)
        optimizer= EMAOptimizer(optimizer=optimizer,device=torch.device('cuda'))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=self.trainer.max_steps,
            num_warmup_steps=int(self.trainer.max_steps * 0.2),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

import os
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import wandb
from PIL import Image
from typing import Callable

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import segmentation_models_pytorch as smp
from lion_pytorch import Lion
from sklearn.model_selection import KFold
from swin_base import register_encoder, register_encoder2

import warnings
warnings.filterwarnings('ignore')

def get_training_augmentation() -> Callable:
    train_transform = [
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.25, border_mode=0),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=0.25),
                A.HueSaturationValue(p=0.25),
            ],
            p=0.3,
        ),
        A.OneOf(
            [
                A.RandomBrightness(p=0.25, limit=0.35),
                A.ColorJitter(p=0.25),
            ],
            p=0.3,
        ),
        A.OneOf(
            [
                A.ChannelShuffle(p=0.25),
                A.RandomGamma(p=0.25),
            ],
            p=0.3,
        ),
        A.Normalize(),
        ToTensorV2()
    ]
    return A.Compose(train_transform)

def get_validation_augmentation() -> Callable:
    valid_transform = [
        A.Normalize(),
        ToTensorV2()
    ]
    return A.Compose(valid_transform)

def dice_score(prediction: np.array, ground_truth: np.array, smooth=1e-7) -> float:
    intersection = torch.sum(prediction * ground_truth)
    return (2.0 * intersection + smooth) / (torch.sum(prediction) + torch.sum(ground_truth) + smooth)


class SatelliteDataset(Dataset):
    def __init__(self, img_pa, transform: Callable=None, infer: bool=False):
        self.img_pa = img_pa
        self.transform = transform
        self.infer = infer

    def __len__(self) -> int:
        return len(self.img_pa)

    def __getitem__(self, idx: int):
        img_path = '../open' + self.img_pa[idx].split('.')[1] + '.png'
        image = Image.open(img_path)
        image = np.array(image)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_path = self.img_pa[idx].split('/')[2]
        mask = np.load(f'../open/train_mask/' + mask_path.split('.')[0] + '.npy')

        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask
    

class SatelliteDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, batch_size, num_workers):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        kf = KFold(n_splits=10, shuffle=True, random_state=36)
        folds = list(kf.split(self.data))
        train_idx, val_idx = folds[0]
        self.train_data = SatelliteDataset([self.data.iloc[i, 1] for i in train_idx], transform= get_training_augmentation(), infer=False)
        self.val_data = SatelliteDataset([self.data.iloc[i, 1] for i in val_idx], transform=get_validation_augmentation(), infer=False)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)


class SatelliteModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = smp.create_model(
            arch=f'{args.model}',
            encoder_name=f"{args.encoder_model}",
            in_channels=3,
            classes=1,
)
        self.loss_fn = smp.losses.DiceLoss(mode="binary")
        self.train_step_outputs = []
        self.val_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, idx):
        image, mask = batch
        pred_mask = self.forward(image)
        mask = mask.unsqueeze(dim=1)
        loss = self.loss_fn(pred_mask, mask)
        
        pred_mask = torch.sigmoid(pred_mask).detach()
        pred_mask = (pred_mask > args.threshold).to(torch.uint8)
        
        mask = mask.detach()
        
        score = 0
        if torch.sum(mask) > 0 or torch.sum(pred_mask) > 0:
            score = dice_score(pred_mask, mask)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_DiceScore", score, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if idx % 40 == 0:
            images, mask = batch
            image = images[0, ...].detach().cpu().numpy()
            image = image.transpose(1, 2, 0)
            mask = mask[0, ...].detach().cpu().numpy()
            pred_mask = pred_mask[0, ...].detach().cpu().numpy()
            self.logger.experiment.log({
                'train_rgb': wandb.Image(image, caption='Input-image'),
                'train_mask': wandb.Image(mask, caption='mask'),
                'train_pred': wandb.Image(pred_mask, caption='pred_mask')
            })
        self.train_step_outputs.append(loss)
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_step_outputs).mean()
        logs = {"fold_train_loss": avg_loss}
        self.train_step_outputs.clear()
        return {"log": logs, "progress_bar": logs}

    def validation_step(self, batch, idx):
        image, mask = batch
        pred_mask = self.forward(image)
        mask = mask.unsqueeze(dim=1)
        loss = self.loss_fn(pred_mask, mask)
        
        pred_mask = torch.sigmoid(pred_mask).detach() 
        pred_mask = (pred_mask > args.threshold).to(torch.uint8)
        
        mask = mask.detach()

        score = 0
        if torch.sum(mask) > 0 or torch.sum(pred_mask) > 0:
            score = dice_score(pred_mask, mask)
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_DiceScore", score, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.val_step_outputs.append(loss)

        if idx % 40 == 0:
            images, mask = batch
            image = images[0, ...].detach().cpu().numpy()
            image = image.transpose(1, 2, 0)
            mask = mask[0, ...].detach().cpu().numpy()
            pred_mask = pred_mask[0, ...].detach().cpu().numpy()
            self.logger.experiment.log({
                'val_rgb': wandb.Image(image, caption='Input-image'),
                'val_mask': wandb.Image(mask, caption='mask'),
                'val_pred': wandb.Image(pred_mask, caption='pred_mask')
            })
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_step_outputs).mean()
        logs = {"fold_val_loss": avg_loss}
        self.val_step_outputs.clear()
        return {"log": logs, "progress_bar": logs}
    
    def configure_optimizers(self):
        optimizer = Lion(self.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=0)
        if args.scheduler:
            return [optimizer] , [scheduler]
        return optimizer

if __name__ =='__main__':

    pl.seed_everything(36, workers=True)
    wandb_logger = WandbLogger(project="dacon_satellite_new")
    register_encoder()
    register_encoder2()

    parser = argparse.ArgumentParser()

    parser.add_argument('--valid', action='store_true', default=True)
    parser.add_argument('--valid_rate',  type=float, default=0.9)
    parser.add_argument('--model', type=str, default='Linknet')
    parser.add_argument('--encoder_model', type=str, default='timm-efficientnet-b7')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epoch', type=int, default=80)
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--scheduler', action='store_true', default=True)
    parser.add_argument('--checkpoint_name', type = str, default = '')
    args = parser.parse_args()

    now = datetime.now()
    now = now.strftime('%m-%d_%H-%M-%S')

    os.makedirs(f'../checkpoints', exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
            dirpath=f"../checkpoints", 
            save_top_k=-1, 
            every_n_epochs= 1,
            monitor="val_DiceScore",
            mode='max',
            filename="{epoch}_{train_loss:2f}_{train_DiceScore:2f}"
    )

    n_cpu = os.cpu_count() // 2
    data_module = SatelliteDataModule(csv_file='../open/train.csv', batch_size= args.batch_size, num_workers= n_cpu)
    satellite = SatelliteModel.load_from_checkpoint(f"../checkpoints/{args.checkpoint_name}.ckpt")
    early_stopping = EarlyStopping(monitor="val_DiceScore", min_delta=0.00, patience=25, verbose=False, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        devices=[0,1,2,3],
        strategy='ddp_find_unused_parameters_true',
        callbacks=[checkpoint_callback , early_stopping, lr_monitor],
        precision=16,
        max_epochs=args.max_epoch,
        logger=wandb_logger,
    )

    trainer.fit(
        satellite, 
        datamodule=data_module
    )
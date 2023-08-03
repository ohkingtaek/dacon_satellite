import os
import argparse
import cv2
import pandas as pd
import numpy as np
import wandb
from PIL import Image
from typing import Callable

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import segmentation_models_pytorch as smp

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Argparse Tutorial')

parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--encoder_model', type=str, default='timm-efficientnet-b7')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--threshold', type=float, default=0.3)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--submit_name', type=str, default='linknet_1024base')
parser.add_argument('--checkpoint_name', type=str, default='')

args = parser.parse_args()

pl.seed_everything(36, workers=True)
wandb_logger = WandbLogger(project="dacon_satellite_test")

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_test_augmentation() -> Callable:
    test_transform = [
        A.Resize(height=args.img_size, width=args.img_size),
        A.Normalize(),
        ToTensorV2()
    ]
    return A.Compose(test_transform)


class SatelliteDataset(Dataset):
    def __init__(self, csv_file: str, transform: Callable=None, infer: bool=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_path = 'open' + self.data.iloc[idx, 1].split('.')[1] + '.png'
        image = Image.open(img_path)
        image = np.array(image)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

n_cpu = (os.cpu_count() // 2)
test_dataset = SatelliteDataset(csv_file='./open/test.csv', transform=get_test_augmentation())
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_cpu, drop_last=True, pin_memory=True)


class SatelliteModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = smp.create_model(
            arch=f'{args.model}',
            encoder_name=args.encoder_model,
            in_channels=3,
            classes=1,
        )
        self.loss_fn = smp.losses.DiceLoss(mode='binary')
        self.result = []

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, idx):
        image = batch
        pred_masks = self.forward(image)
        pred_masks = torch.sigmoid(pred_masks).detach().cpu().numpy()
        
        for pred_mask in pred_masks:
            pred_mask = (pred_mask > args.threshold).astype(np.uint8)
            for mask in pred_mask:
                mask_rle = rle_encode(mask)
                if mask_rle == '':
                    self.result.append(-1)
                else:
                    self.result.append(mask_rle)
            
        if idx % 50 == 0:
            mask = np.squeeze(mask).astype(np.uint8)
            image = image[0, ...].detach().cpu().numpy()
            image = image.transpose(1, 2, 0)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            for contour in contours:
                cv2.fillPoly(mask, [contour], color=1)
            mask = mask * 255

            self.logger.experiment.log({
                'rgb': wandb.Image(image, caption='Input-image'),
                'mask': wandb.Image(mask, caption='mask'),
                'pred_mask': wandb.Image(pred_mask, caption='mask')
            })

    def on_predict_end(self) -> None:
        submit = pd.read_csv('./open/sample_submission.csv')
        submit['mask_rle'] = self.result
        submit.to_csv(f'./results/{args.submit_name}.csv', index=False)

satellite = SatelliteModel.load_from_checkpoint(f"./checkpoints/{args.checkpoint_name}.ckpt")

trainer = pl.Trainer(
    devices=[0],
    logger=wandb_logger,
)

trainer.predict(
    satellite, 
    test_dataloader,
)
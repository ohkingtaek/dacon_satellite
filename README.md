# Dacon Satellite Semantic Segmentation
https://dacon.io/competitions/official/236092/overview/description

## Set Environment
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

```
├─checkpoints
├─open
│  ├─test_img
│  ├─train_img
│  └─train_mask
├─results
├─script
└─tool
```

## Download Weights
https://drive.google.com/file/d/1QKXYf73BdY2C6WZzWNRIeI-bIQ4Pt7S3/view?usp=sharing

## Make Pretrain Model
```
sh pretrain.sh
```

## Train
```
sh train.sh
```

## Test
```
sh test.sh
```

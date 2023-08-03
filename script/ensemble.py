import pandas as pd
import numpy as np
from tqdm import tqdm


UnetPlusPlus_pretrained = pd.read_csv('./upp_ptt.csv')
linknet_pretrained = pd.read_csv('./linknet_1024base.csv')
Unet = pd.read_csv('./Unet1024.csv')
sample_submission = pd.read_csv('./sample_submission.csv')


def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

Prediction = []

for idx in tqdm(range(len(UnetPlusPlus_pretrained))):
    model_UnetPlusPlus = np.array(rle_decode(UnetPlusPlus_pretrained['mask_rle'][idx], (224, 224)))
    model_Linknet = np.array(rle_decode(linknet_pretrained['mask_rle'][idx], (224, 224)))
    model_Unet = np.array(rle_decode(Unet['mask_rle'][idx], (224, 224)))
    
    results = []
    for i in range(len(model_UnetPlusPlus)):
        temp_res = []
        for j in range(len(model_UnetPlusPlus)):
            best_model = model_UnetPlusPlus[i][j]
            second_model = model_Linknet[i][j]
            third_model = model_Unet[i][j]
            
            result = best_model + second_model + third_model
            if results >= 2:
                temp_res.append(1)
            else:
                temp_res.append(0)
        results.append(temp_res)
    rle = rle_encode(np.array(results))
    if rle == '':
        Prediction.append(-1)
    else:
        Prediction.append(rle)

sample_submission['mask_rle'] = Prediction
sample_submission.to_csv('./voting_3models.csv', index=False)
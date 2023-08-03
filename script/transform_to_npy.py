import numpy as np
import pandas as pd
import cv2

data = pd.read_csv('../open/train.csv')

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

for idx in range(len(data)):
    img_path = './open' + data.iloc[idx, 1].split('.')[1] + '.png'
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_rle = data.iloc[idx, 2]
    mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

    data_path = data.iloc[idx, 1].split('/')[2]
    np.save(f'./open/train_mask/' + data_path.split('.')[0] + '.npy', mask)
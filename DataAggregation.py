import os
import numpy as np
from PIL import Image

BASE_PATH = '/usr/xtmp/DS_GRSS/CS474/Dataset/200SquareMaskData'
TRAIN_PATH = os.path.join(BASE_PATH, 'Train')
SAVE_PATH = '/usr/xtmp/DS_GRSS/CS474/Pytorch-UNet/data'
IMG_PATH = os.path.join(SAVE_PATH, 'imgs')
MASK_PATH = os.path.join(SAVE_PATH, 'masks')

def AggregateAndStore():
    for fileName in sorted(os.listdir(TRAIN_PATH)):
        try:
            imgID = int(fileName)
        except:
            continue
        print("Processing Data ID {}".format(fileName))
        currDir = os.path.join(TRAIN_PATH, fileName)
        channels = []
        for arrays in sorted(os.listdir(currDir)):
            print(arrays)
            currImage = np.load(os.path.join(currDir, arrays))
            if("groundTruth.npy" in fileName):
                print("here")
                gtPath = os.path.join(MASK_PATH, "{}.npy".format(imgID))
                np.save(gtPath, currImage)
            else:
                channels.append(currImage)
        channels = np.array(channels)
        imgPath = os.path.join(IMG_PATH, "{}.npy".format(imgID))
        np.save(imgPath, channels)

AggregateAndStore()


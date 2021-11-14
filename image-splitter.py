import numpy as np
from os import listdir
from PIL import Image
from math import sqrt

# Split square image into n images.
def split(img, n):
    h = img.shape[0]
    step = int(h / sqrt(n))
    images = []

    x1, y1 = 0, step
    for i in range(int(sqrt(n))):
        x2, y2 = 0, step
        for j in range(int(sqrt(n))):
            images.append(img[x1 : y1, x2 : y2])
            x2 += step
            y2 += step
        x1 += step
        y1 += step

    return images

dirDataset = '/home/cam/Documents/Texture-Analysis/train/'
dirTextures = sorted(listdir(dirDataset))

for i in range(13):
    dirImage = dirDataset + dirTextures[i] + '/' + listdir(dirDataset + dirTextures[i])[0]
    img = np.asarray(Image.open(dirImage).convert('L')).astype(np.uint8)
    cropped = split(img, 16)
    for j in range(len(cropped)):
        temp = Image.fromarray(cropped[j])
        temp.save(f'{dirDataset}{dirTextures[i]}/{dirTextures[i]}_000_{j+1}.tiff')
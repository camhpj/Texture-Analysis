import numpy as np
from PIL import Image
import skimage
from skimage.color import rgb2gray
from scipy.io import savemat
from os import listdir


def findMSCM(img, r, maxGL=256, numBins=32):
    #--Allocate space for MSCM.
    MSCM = np.zeros((numBins, numBins)).astype(np.float64)

    #--Create logical mask for circle of radius r.
    circleWidth = (2 * r) + 1
    [X, Y] = np.meshgrid(np.arange(1, (circleWidth + 1)), np.arange(1, (circleWidth + 1)))
    distFromCenter = np.sqrt(np.power(X - (r + 1), 2) + np.power(Y - (r + 1), 2))
    onPixels = np.abs(distFromCenter - r) < 0.5

    #--Remove bottom half of circle from mask.
    f = Y - (r+1)
    onPixels[f > 0] = 0
    onPixels[r, 0] = 0

    #--Change pixels with value 0 to 1 (prevents reshape errors when collecting entries).
    img[img == 0] = 1
    rows, cols = img.shape[0], img.shape[1]
    img = np.pad(img, r, constant_values=-1)

    #--Collect all windows and centers (binned in place) in the image.
    windows = skimage.util.view_as_windows(img, circleWidth)
    centers = np.ceil(np.divide(np.squeeze(windows[:, :, r, r]), (maxGL / numBins))).astype(np.int64)

    #--Collect and normalize neighbor values. Also, set collected values from padding to 0.
    pixelsNeeded = [4, 6, 8, 16]
    windows_onPixels = np.multiply(windows, onPixels)
    values = np.reshape(windows_onPixels[windows_onPixels != 0], [rows, cols, pixelsNeeded[r - 1]])
    values[values == -1] = 0
    values = np.ceil(np.divide(windows_onPixels, (maxGL / numBins))).astype(np.int64)
    values = np.reshape(values, [rows, cols, (pixelsNeeded[r - 1] * 2) + 1])

    #--Add values to MSCM bin by bin.
    for i in range(0, 32):
        #--Collect entries for current bin value. Remove entries that are equal to 0.
        entries = np.where(centers == i+1)
        entries = values[entries[0][:], entries[1][:], :]
        entries = entries[entries != 0]
        
        #--Bin values (33 bins but 0th bin is not used).
        entries = np.bincount(entries, minlength=33)

        #--Find indices of values to enter.
        nonZero = np.where(entries != 0)[0]

        #--Subtract 1 to match MSCM indices to entries indices. Add entries to MSCM.
        MSCM[i, np.subtract(nonZero,1)] = MSCM[i, np.subtract(nonZero, 1)] + entries[nonZero]

    return MSCM

if __name__ == '__main__':
    #--Dataset parameters.
    dirDataset = '' # Dataset file path.
    dirTextures = sorted(listdir(dirDataset))
    fileType = '' # Image file type.

    #--Output parameters.
    dirSave = '' # Directory to save MSCM to.
    matName = '' # MATLAB variable name.
    numTextures = 0
    numImages = 0 # Number of images per texture.
    imageDims = [0, 0]
    angle = '' # Angle of image rotation.
    r = [] # List of radii to calculate MSCM for.

    #--Allocate space for images (xDim, yDim, numTextures, numImages/Texture).
    images = np.zeros((imageDims[0], imageDims[1]))

    print("Reading images")
    for i in range(numTextures):
        for j in range(numImages):
            images[:, :, i, j] = np.array(Image.open(dirDataset + dirTextures[i] + '/' + dirTextures[i] + f'_{angle}_{j+1}.{fileType}')).astype(np.uint8)
    print("Done")

    #--Allocate space for MSCM.
    MSCM = np.zeros((32, 32, numTextures, numImages, len(r)))

    print("Calculating MSCM")
    for i in range(numTextures):
        for j in range(numImages):
            for k in len(r):
                MSCM[:, :, i, j, k] = findMSCM(images[:, :, i, j], r[k])
        print(f"Finished texture #{i+1}")
    print("Done")

    mdic = {matName: MSCM}
    savemat(dirSave, mdict=mdic)
    print(f"MSCM saved to: {dirSave}")
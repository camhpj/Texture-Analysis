import numpy as np
from PIL import Image
import skimage
from skimage.color import rgb2gray
from scipy.io import savemat
from os import listdir


def findMSCM(img, maxGL, numBins, r):
    #--Allocate space for MSCM.
    MSCM = np.zeros((numBins, numBins)).astype(np.float64)

    #--Create logical mask for circle or radius r.
    circleWidth = (2 * r) + 1
    [X, Y] = np.meshgrid(np.arange(1, (circleWidth + 1)), np.arange(1, (circleWidth + 1)))
    distFromCenter = np.sqrt(np.power(X - (r + 1), 2) + np.power(Y - (r + 1), 2))
    onPixels = np.abs(distFromCenter - r) < 0.5

    #--Change pixels with value 0 to 1 (prevents reshape errors when collecting entries).
    img[img == 0] = 1
    rows, cols = img.shape[0], img.shape[1]

    #--Collect all windows and centers in the image.
    windows = skimage.util.view_as_windows(img, circleWidth)
    
    #--Collect and normalize centers.
    centers = np.ceil(np.divide(np.squeeze(windows[:, :, r, r]), (maxGL / numBins))).astype(np.int64)

    #--Collect and normalize values.
    pixelsNeeded = [8, 12, 16, 32]
    windows_onPixels = np.multiply(windows, onPixels)
    values = np.ceil(np.divide(windows_onPixels, (maxGL / numBins))).astype(np.int64)
    values = np.reshape(values[values != 0], [rows - (2 * r), cols - (2 * r), pixelsNeeded[r - 1]])

    #--Add values to MSCM bin by bin.
    for i in range(0, 32):
        #--Collect entries for current bin value.
        entries = np.where(centers == i+1)
        n = len(entries[0])
        entries = values[entries[0][:], entries[1][:], :]

        #--Create IDs so values can bin with respect to window.
        id = entries + (33*np.arange(n))[:,None]
        
        #--Bin values.
        entries = np.bincount(id.ravel(), minlength=33*n).reshape(-1, 33)
        entries = np.sum(entries, axis=0)

        #--Find indices of values to enter.
        nonZero = np.where(entries != 0)[0]

        #--Subtract 1 to match MSCM indices to entries indices. Add entry to MSCM.
        MSCM[i, np.subtract(nonZero,1)] = MSCM[i, np.subtract(nonZero, 1)] + entries[nonZero]

    return MSCM


if __name__ == '__main__':
    #--Necessary parameters
    maxGL = 256
    numBins = 32

    imgDir = '/home/cam/Documents/Fall 2021/Texture-Analysis/train/D9/D9_000_1.tiff'
    img = np.array(Image.open(imgDir)).astype(np.uint8)
    MSCM = np.zeros((32, 32, 4))

    r = [1, 2, 3, 4]
    for k in range(3):
        MSCM[:, :, k] = findMSCM(img, maxGL, numBins, r[k])

    
    mdic = {"MSCM_circle_method": MSCM}
    savemat('/home/cam/Documents/Fall 2021/Texture-Analysis/MSCM Files/MSCM_circle_method.mat', mdict=mdic)
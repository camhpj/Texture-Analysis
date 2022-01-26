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

    #--Dataset file path.
    dirDataset = '<dataset filepath>'
    dirTextures = sorted(listdir(dirDataset))
    angle = '<rotation angle of images (for file naming)'

    #--Allocate space for images (dimx, dimy, # of textures, # of images per texture). 
    #--Must be modified for the correct image dimensions, texture class number, and number of images per class.
    images = np.zeros((128, 128, 13, 16))

    print("Reading images.")

    for i in range(13):
        for j in range(16):
            images[:, :, i, j] = np.array(Image.open(dirDataset + dirTextures[i] + '/' + dirTextures[i] + f'_{angle}_{j+1}.tiff')).astype(np.uint8)

    print("Done.")
    
    #--Allocate space for MSCM. (MSCM, MSCM, texture, image, r)
    MSCM = np.zeros((32, 32, 13, 16, 4))

    r = [1, 2, 3, 4]
    for i in range(13):
        for j in range(16):
            for k in range(4):
                MSCM[:, :, i, j, k] = findMSCM(images[:, :, i, j], maxGL, numBins, r[k])
        print(f'Finished texture #{i+1}')

    
    mdic = {"brodatzMSCM_test_200": MSCM}
    savemat('/home/cam/Documents/Texture-Analysis/MSCM Files/brodatz-mscm-test-200.mat', mdict=mdic)

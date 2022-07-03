# Texture-Analysis
This goal of this project is to develop and test a new method for obtaining rotation invariant texture measures.

## Datasets
### Brodatz
This dataset consists of 13 texture classes each containing a single 512x512 image. To increase the sample size for image classification the original images are subdivided into 16 128x128 non-overlapping images.

The train folder contains the 16 unrotated image samples for each of the 13 classes and is used for training the classifier.
The test folders (test-rotated-x) contain the 16 image samples at the specified rotation for each class.

## The Multi-Scale Co-Occurence Matrix
The multi-scale co-occurrence matrix, constructed similarly to the GLCM, is parametrized only by a radius r. Unlike the GLCM, the MSCM considers the entire circle of pixels at a radius r and not just 4 neighbors. The figure below shows this neighborhood for r=1 which consists of 8 pixels. 

![MSCM Neigborhood for r=1](https://github.com/camhpj/Texture-Analysis/blob/main/mscm-neighborhood.jpg)

As the radius increases the number of pixels counted also increases (12 for r=2, 16 for r=3, and 32 for r=4). Due to the similarity between the two, the MSCM can be thought of as a stack of GLCMs where each slice is a GLCM for a particular radius. The MSCM contains information from every discrete direction as well as at multiple distances. Under ideal rotations (90 and 180) the MSCM calculated will be identical to the MSCM calculated from the unrotated image. The goal of this more expensive calculation is to reduce the amount by which features calculated from a texture change when the texture is rotated.

An MSCM in the code is defined as a 5 dimensional array (GLCM, GLCM, texture class, image #, radius #). This is done to make handling the data structure more streamlined. The first two dimensions contain a co-occurence matrix. The next 3 dimensions are used to specify the GLCM to be operated on using the class, image, and radius numbers.

## Using this library
The calculation of texture features is performed using the extract-features.ipynb Jupyter notebook. The files calculate.py and utils.py contain the funtions that are run using this notebook. To configure the script the variables path, radii, numGL, and numBins must be modified. The path variable contains the path to a dataset as a string. Datasets must be structured as is seen in the brodatz folder. The variable radii is a list of which radii to calculate the MSCM for. The large the radius the longer the compute time as more pixels must be included in each calculation. Running the notebook will yield a mat files containing features for each folder of images.

Classification with K-Nearest Neighbors is performed using classification_harness.mlx. utils.m contains a function which is used by the notebook to build a list of labels needed for classification. The paths to the train and test features must be updated. In addition radii, numClasses, and numImages should reflect the dataset and the radii for which the MSCM was calculated. numRotations should be the number of rotations performed (or folders of images to perform classification predictions on.

## Notes
The file utils.py also contains one unused function for splitting images. This is useful for increasing sample size when a small number of high resolution textures are avaiable. This function currently only works for square images.

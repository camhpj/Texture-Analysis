# Texture-Analysis
All of the files associated with the multi-scale co-occurence matrix and image texture classifications.

The train folder contains 16 images (cropped from the original unrotated image) for the 13 texture classes.
The original textures are 512 by 512. The cropped images are 128 by 128.
Sorted textures order -> D112, D12, D15, D16, D19, D24, D29, D38, D68, D84, D9, D92, D94.

brodatz-train-mscm is a defined by (MSCM, MSCM, texture class, image #, radius #).
brodatzFeatures is contains contrast, correlation, energy, and homogeneity for each image (feature, texture, image, radius).
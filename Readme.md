# Image Clustering Code

This repository contains code to cluster line graph image patches.

# Setup

Download the serialized files from
[HERE](https://drive.google.com/drive/folders/1U0WKZV4r7s9eIoFGdEZzqtdIBGh4ATEb?usp=sharing)

# Structure

Repository Files:

-   `main.py`: Python code to cluster the image patches
-   `demo.ipynb`: Jupyter notebook demonstrating the closest clusters to
    3 test images:
    -   `line-test-1.jpg`
    -   `line-test-2.jpg`
    -   `line-test-3.jpg`
-   `im/`: Training images (Ed's dataset, needs to be added)

[Serialized
Files](https://drive.google.com/drive/folders/1U0WKZV4r7s9eIoFGdEZzqtdIBGh4ATEb?usp=sharing):

-   `features_V0.pickle`: Serialized
    [HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)
    feature vectors for each patch
-   `km_model_V0.pickle`: Serialized K-Nearest Neighbors classifier
-   `labeled_imgs_V0.pickle`: Serialized labeled image patches
-   `contexts_V0.pickle`: "sentences" used for training word2vec model
-   `wv_model_V0.pickle`: trained word2vec model

To get the classified images patches you can just run:

``` python
import pickle
with open("labeled_imgs.pickle", "rb") as serialized_object:
    labeled_imgs = pickle.load(serialized_object)
```

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
-   `im/`: Training images

[Serialized
Files](https://drive.google.com/drive/folders/1U0WKZV4r7s9eIoFGdEZzqtdIBGh4ATEb?usp=sharing):

-   `features.pickle`: Serialized
    [HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)
    feature vectors for each patch
-   `model.pickle`: Serialized K-Nearest Neighbors classifier
-   `labeled_imgs.pickle`: Serialized labeled image patches

To get the classified images patches you can just run:

``` python
import pickle
with open("labeled_imgs.pickle", "rb") as serialized_object:
    labeled_imgs = pickle.load(serialized_object)
```

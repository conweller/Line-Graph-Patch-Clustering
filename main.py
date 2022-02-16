import imghdr
import os
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.cluster import KMeans
from skimage.feature import hog
from skimage.util import view_as_windows

IMG_SIZE = 448
WINDOW_SIZE = 32
STRIDE = 32
TRAIN_SIZE = 998


def open_img(filename, color="L", size=(IMG_SIZE, IMG_SIZE)):
    """return grayscale image matrix"""
    return np.array(Image.open(filename).convert(color).resize(size))


def get_img_sections(img):
    """Covert image to image sections"""
    windowed = view_as_windows(img, window_shape=(
        WINDOW_SIZE, WINDOW_SIZE), step=STRIDE)
    (nx, ny, w, h) = windowed.shape
    return windowed.reshape(nx * ny, w, h)


def get_features(section):
    """Given section return the slope of it's edges"""
    return hog(section)


def pickle_load_or_create(filename, default_fn):
    """load and return pickled object, or execute default function,
    pickle the result and return the result
    """
    if os.path.isfile(filename):
        with open(filename, "rb") as serialized_object:
            return pickle.load(serialized_object)
    else:
        object_ = default_fn()
        with open(filename, "wb") as serialized_object:
            pickle.dump(object_, file=serialized_object)
        return object_


def get_all_features(imgs):
    """Return all of HOG features for the given images"""
    features = []
    for _ in range(TRAIN_SIZE):
        img = next(imgs)
        for section in get_img_sections(img):
            features.append(get_features(section))
    return np.array(features)


def label_imgs(labels):
    """Return iamge represented by the labels"""
    return labels.reshape(TRAIN_SIZE, IMG_SIZE // WINDOW_SIZE, IMG_SIZE // WINDOW_SIZE)


filenames = list(filter(imghdr.what, map(
    lambda imgname: Path("im") / imgname, os.listdir("im"))))

imgs = map(open_img, filenames)

features = pickle_load_or_create(
    "features.pickle", lambda: np.array(get_all_features(imgs)))

model = pickle_load_or_create(
    "model.pickle", lambda: KMeans(n_clusters=1000).fit(features))

labeled_imgs = pickle_load_or_create(
    "labeled_imgs.pickle", lambda: label_imgs(model.labels_))

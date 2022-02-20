import imghdr
import os
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.cluster import KMeans
from skimage.feature import hog
from skimage.util import view_as_windows
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
from matplotlib import pyplot as plt

IMG_SIZE = 448
WINDOW_SIZE = 32
STRIDE = 32
TRAIN_SIZE = 998
CONTEXT_SIZE = 5


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


def get_labeled_imgs(labels):
    """Return iamge represented by the labels"""
    return labels.reshape(TRAIN_SIZE, IMG_SIZE // WINDOW_SIZE, IMG_SIZE // WINDOW_SIZE)


def get_contexts(labeled_imgs):
    """Get Context windows for each pixels (i.e. convert a labeled imgs into
    the training input for word2vec)
    """
    (n, w, _) = labeled_imgs.shape
    n_contexts_per_img = ((w+1) - CONTEXT_SIZE)**2
    contexts = np.empty((n * n_contexts_per_img, CONTEXT_SIZE**2))
    for idx, img in enumerate(labeled_imgs):
        contexts[idx*n_contexts_per_img:idx*n_contexts_per_img+n_contexts_per_img] = view_as_windows(img, window_shape=(
            CONTEXT_SIZE, CONTEXT_SIZE)).reshape(n_contexts_per_img, CONTEXT_SIZE**2)
    return contexts.astype(int)


def get_wv_model(contexts):
    """Given training contexts produce word2vec model"""
    wv_model = Word2Vec(min_count=1, window=8)
    wv_model.build_vocab(contexts.astype(str).tolist())
    wv_model.train(contexts.astype(str).tolist(),
                   total_examples=wv_model.corpus_count, epochs=wv_model.epochs)
    return wv_model


def show_cluster(km_model, cluster_idx, nrows=5, ncols=5):
    """Show a sampling of a given cluster"""
    cluster_example_idxs = np.where(km_model.labels_ == cluster_idx)[0]
    _, axarr = plt.subplots(nrows, ncols, figsize=(15, 10))
    for ax_idx, feature_idx in enumerate(cluster_example_idxs[:nrows*ncols]):
        filename_idx = feature_idx // ((IMG_SIZE // WINDOW_SIZE)**2)
        axarr[ax_idx // nrows, ax_idx % ncols].imshow(get_img_sections(open_img(
            filenames[filename_idx]))[feature_idx % ((IMG_SIZE // WINDOW_SIZE)**2)])
    plt.show()


def most_similar_clusters(km_model, cluster_idx, metric=cosine_similarity, topn=10):
    """Shows the most similar clusters to a given cluster (assumes the metric
    is a distance metric unless it is cosine_similarity)"""
    scores = metric([km_model.cluster_centers_[cluster_idx]],
                    km_model.cluster_centers_)[0]
    if metric == cosine_similarity:
        best = np.argsort(scores)[-(topn+1):-1][::-1]
    elif metric in [euclidean_distances, manhattan_distances]:
        best = np.argsort(scores)[1:(topn+1)]
    else:
        return None
    return list(zip(best, scores[best]))


def map_similarity_to_img(wv_model, labeled_img, cluster_idx):
    """Creates new image array of word2vec similarity scores to the query idx"""
    (w, h) = labeled_img.shape
    new_image = np.empty((w, h))
    for i in range(w):
        for j in range(h):
            new_image[i, j] = wv_model.wv.similarity(
                str(cluster_idx), labeled_img[i, j])
    return new_image


filenames = list(filter(imghdr.what, map(
    lambda imgname: Path("im") / imgname, os.listdir("im"))))

imgs = map(open_img, filenames)

features = pickle_load_or_create(
    "features.pickle", lambda: np.array(get_all_features(imgs)))

km_model = pickle_load_or_create(
    "km_model.pickle", lambda: KMeans(n_clusters=1000).fit(features))

labeled_imgs = pickle_load_or_create(
    "labeled_imgs.pickle", lambda: get_labeled_imgs(km_model.labels_))

contexts = pickle_load_or_create(
    "contexts.pickle", lambda: get_contexts(labeled_imgs))

wv_model = pickle_load_or_create(
    "wv_model.pickle", lambda: get_wv_model(contexts))

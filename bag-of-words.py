from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from main import open_img, km_model, wv_model,get_labeled_imgs, get_all_features, embedded_img
from pathlib import Path
import imghdr
import os
import numpy as np

with open("labels.txt") as inf:
    labels = dict(map(str.split, inf.readlines()))

filenames = list(filter(imghdr.what, map(
    lambda imgname: Path("im") / imgname, os.listdir("im"))))

imgs = map(open_img, filenames)

features = get_all_features(imgs)

labeled_imgs = get_labeled_imgs(km_model.predict(features))

n_imgs, h, w = labeled_imgs.shape

embedded_imgs = np.empty((n_imgs, h, w, 100))
summed_wv_featueres = np.empty((n_imgs, 100))

for i in range(n_imgs):
    embedded_imgs[i] = embedded_img(wv_model, labeled_imgs[i])
    summed_wv_featueres[i] = embedded_imgs[i].sum(axis=(0,1))



X_train_clusters = labeled_imgs[:850].reshape(850, 14 * 14)
X_test_clusters = labeled_imgs[850:].reshape(148, 14 * 14)

X_train_embed = embedded_imgs[:850].reshape(850, 14 * 14 * 100)
X_test_embed = embedded_imgs[850:].reshape(148, 14 * 14 * 100)

X_train_hog = features.reshape(998,196*324)[:850].reshape(850, 196*324)
X_test_hog = features.reshape(998,196*324)[850:].reshape(148, 196*324)

y_train = np.array([int(labels[str(fname)]) for fname in filenames[:850]])
y_test = np.array([int(labels[str(fname)]) for fname in filenames[850:]])

scalar = StandardScaler()
clf = LogisticRegression()


pipeline_clusters = Pipeline([('scaling', scalar), ('logistic', clf)])
pipeline_clusters.fit(X_train_clusters, y_train)
prediction_clusters=pipeline_clusters.predict(X_test_clusters)
print(metrics.classification_report(y_test, prediction_clusters))
print(metrics.confusion_matrix(y_test, prediction_clusters))
print(metrics.accuracy_score(y_test, prediction_clusters))

pipeline_embed = Pipeline([('scaling', scalar), ('logistic', clf)])
pipeline_embed.fit(X_train_embed, y_train)
prediction_embed=pipeline_embed.predict(X_test_embed)
print(metrics.classification_report(y_test, prediction_embed))
print(metrics.confusion_matrix(y_test, prediction_embed))
print(metrics.accuracy_score(y_test, prediction_embed))

pipeline_hog = Pipeline([('scaling', scalar), ('logistic', clf)])
pipeline_hog.fit(X_train_hog, y_train)
prediction_hog=pipeline_hog.predict(X_test_hog)
print(metrics.classification_report(y_test, prediction_hog))
print(metrics.confusion_matrix(y_test, prediction_hog))
print(metrics.accuracy_score(y_test, prediction_hog))

# accuracy_score for clusters: 37.8%
# accuracy_score for embedded: 54%
# accuracy_score for hog: 55%

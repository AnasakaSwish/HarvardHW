from collections import defaultdict
import json
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import os



def capitalize(s):
    s = s.title()
    s = s.replace("Of", "of")
    return s


# census_data["State"] = census_data.state.map(capitalize)
# del census_data["state"]
# census_data['State'] = census_data['State'].replace(abbrev_states_dict)
# census_data.set_index("State", inplace=True)
# smaller_frame = census_data[['educ_coll', 'average_income', 'per_vote']]
#
#
# X_HD = smaller_frame[['educ_coll', 'average_income']].values
# X_HDn = (X_HD - X_HD.mean(axis=0)) / X_HD.std(axis=0)
# educ_coll_std_vec = X_HDn[:, 0]
# educ_coll_std = educ_coll_std_vec.reshape(-1, 1)
# average_income_std_vec = X_HDn[:, 1]
# average_income_std = average_income_std_vec.reshape(-1, 1)
#
#
# X_train, X_test, y_train, y_test = train_test_split(average_income_std, educ_coll_std_vec)
# clf2 = LinearRegression()
# clf2.fit(X_train, y_train)
# predicted_train = clf2.predict(X_train)
# predicted_test = clf2.predict(X_test)
# trains = X_train.reshape(1, -1).flatten()
# tests = X_test.reshape(1, -1).flatten()
#
# plt.plot(average_income_std_vec, clf2.predict(average_income_std))
# plt.scatter(average_income_std_vec, educ_coll_std_vec)
# plt.show()
# from sklearn.linear_model import LogisticRegression
# reg=1.
# data=np.array([[float(j) for j in e.strip().split()] for e in open("./data/chall.txt")])
# temps, pfail = data[:,0], data[:,1]
# clf4 = LogisticRegression(C=reg)
# clf4.fit(temps.reshape(-1,1), pfail)
# tempsnew=np.linspace(20., 90., 15)
# probs = clf4.predict_proba(tempsnew.reshape(-1,1))[:, 1]
# predicts = clf4.predict(tempsnew.reshape(-1,1))
#
# def fit_logistic(X_train, y_train, reg=0.0001, penalty="l2"):
#     clf = LogisticRegression(C=reg, penalty=penalty)
#     clf.fit(X_train, y_train)
#     return clf$
# from sklearn.model_selection import GridSearchCV
#
# def cv_optimize(X_train, y_train, paramslist, penalty="l2", n_folds=10):
#     clf = LogisticRegression(penalty=penalty)
#     parameters = {"C": paramslist}
#     gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds)
#     gs.fit(X_train, y_train)
#     return gs.best_params_, gs.best_score_
#
# def cv_and_fit(X_train, y_train, paramslist, penalty="l2", n_folds=5):
#     bp, bs = cv_optimize(X_train, y_train, paramslist, penalty=penalty, n_folds=n_folds)
#     print "BP,BS", bp, bs
#     clf = fit_logistic(X_train, y_train, penalty=penalty, reg=bp['C'])
#     return clf
#
# clf=cv_and_fit(temps.reshape(-1,1), pfail, np.logspace(-4, 3, num=100))
# print clf
# print pd.crosstab(pfail, clf.predict(temps.reshape(-1,1)), rownames=["Actual"], colnames=["Predicted"])
# print zip(temps,pfail, clf.predict(temps.reshape(-1,1)))

from PIL import Image

STANDARD_SIZE = (322, 137)
def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    if verbose==True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

checks_dir = "./data/images/images/checks/"
dollars_dir = "./data/images/images/dollars/"
def images(img_dir):
    return [img_dir+f for f in os.listdir(img_dir)]
checks=images(checks_dir)
dollars=images(dollars_dir)
images=checks+dollars
labels = ["check" for i in range(len(checks))] + ["dollar" for i in range(len(dollars))]

i0=images[20]
i0m=img_to_matrix(i0)
data = []
for image in images:
    img = img_to_matrix(image)
    img = flatten_image(img)
    data.append(img)

data = np.array(data)
y = np.where(np.array(labels)=="check", 1, 0)

def do_pca(d,n):
    pca = PCA(n_components=n)
    X = pca.fit_transform(d)
    print pca.explained_variance_ratio_
    return X, pca

X5, pca5=do_pca(data,5)
print np.sum(pca5.explained_variance_ratio_)

def normit(a):
    a=(a - a.min())/(a.max() -a.min())
    a=a*256
    return np.round(a)

def getRGB(o):
    size=322*137*3
    r=o[0:size:3]
    g=o[1:size:3]
    b=o[2:size:3]
    r=normit(r)
    g=normit(g)
    b=normit(b)
    return r,g,b

def getNC(pc, j):
    return getRGB(pc.components_[j])

def getMean(pc):
    m=pc.mean_
    return getRGB(m)

def display_from_RGB(r, g, b):
    rgbArray = np.zeros((137,322,3), 'uint8')
    rgbArray[..., 0] = r.reshape(137,322)
    rgbArray[..., 1] = g.reshape(137,322)
    rgbArray[..., 2] = b.reshape(137,322)
    img = Image.fromarray(rgbArray)
    plt.imshow(np.asarray(img))
    ax=plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def display_component(pc, j):
    r,g,b = getNC(pc,j)
    return display_from_RGB(r,g,b)

from sklearn.linear_model import LogisticRegression
def fit_logistic(X_train, y_train, reg=0.0001, penalty="l2"):
    clf = LogisticRegression(C=reg, penalty=penalty)
    clf.fit(X_train, y_train)
    return clf

from sklearn.model_selection import GridSearchCV

def cv_optimize(X_train, y_train, paramslist, penalty="l2", n_folds=10):
    clf = LogisticRegression(penalty=penalty)
    parameters = {"C": paramslist}
    gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds)
    gs.fit(X_train, y_train)
    return gs.best_params_, gs.best_score_

def cv_and_fit(X_train, y_train, paramslist, penalty="l2", n_folds=5):
    bp, bs = cv_optimize(X_train, y_train, paramslist, penalty=penalty, n_folds=n_folds)
    print "BP,BS", bp, bs
    clf = fit_logistic(X_train, y_train, penalty=penalty, reg=bp['C'])
    return clf

from matplotlib.colors import ListedColormap
def points_plot(Xtr, Xte, ytr, yte, clf):
    X=np.concatenate((Xtr, Xte))
    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    f,ax = plt.subplots()
    # Plot the training points
    ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap=cm_bright)
    # and testing points
    ax.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap=cm_bright, marker="s", s=50, alpha=0.9)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.4)
    cs2 = ax.contour(xx, yy, Z, cmap=cm, alpha=.4)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize=14)
    return ax
is_train = np.random.uniform(0, 1, len(data)) <= 0.7
train_x, train_y = data[is_train], y[is_train]
test_x, test_y = data[is_train==False], y[is_train==False]
pca = PCA(n_components=2)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)
logreg = cv_and_fit(train_x, train_y, np.logspace(-4, 3, num=100))
print pd.crosstab(test_y, logreg.predict(test_x), rownames=["Actual"], colnames=["Predicted"])
plt.show()

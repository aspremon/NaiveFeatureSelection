# %% Import packages
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

from naive_feature_selection import NaiveFeatureSelection

# %% Get 20newsgroup data set, cf. "Classification of text documents
# using sparse features" in sklearn doc.
print("Testing NFS ...")
categories = [
        'sci.med',
        'sci.space'
    ]
remove = ('headers', 'footers', 'quotes')
print("Loading 20 newsgroups dataset for categories:")
print(categories if categories else "all")
print()
data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42, remove=remove)
data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42, remove=remove)


# split training / test set
y_train, y_test = data_train.target, data_test.target


# Vectorize data set
print("Extracting features from the training data using a sparse vectorizer")
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
X_test = vectorizer.transform(data_test.data)
print("n_samples: %d, n_features: %d" % X_test.shape)
print()
feature_names = vectorizer.get_feature_names()


# %% Test Naive Feature Selection, followed by l2 SVM
kv = 100  # kv is target number of features
nfs = NaiveFeatureSelection(k=kv)
# Use fit_transform to extract selected features
X_new = nfs.fit_transform(X_train, y_train)
# Train SVM
clfsv = LinearSVC(random_state=0, tol=1e-5)
clfsv.fit(X_new, y_train == 1)
# Test performance
X_testnew = nfs.transform(X_test)
y_pred_NFS = clfsv.predict(X_testnew)
score_nfs = metrics.accuracy_score(y_test == 1, y_pred_NFS)
print("NFS accuracy:\t%0.3f" % score_nfs)
print("")

# List selected features
print('Space features:')
print([feature_names[np.nonzero(nfs.mask_)[0][i]]
       for i in range(kv) if clfsv.coef_[0][i] >= 0])
print()
print('Med features:')
print([feature_names[np.nonzero(nfs.mask_)[0][i]] for i in range(kv) if clfsv.coef_[0][i]<0])
print("")


# %% Use pipeline instead
from sklearn.pipeline import Pipeline
clf = Pipeline([
  ('feature_selection', NaiveFeatureSelection(k=100)),
  ('classification', LinearSVC())
])
clf.fit(X_train, y_train)
y_pred_pp = clf.predict(X_test)
score_pp = metrics.accuracy_score(y_test == 1, y_pred_pp)
print("Pipeline accuracy:\t%0.3f" % score_pp)
print("")


# %% Cross validate to get best k
from sklearn.model_selection import GridSearchCV
parameters = {'feature_selection__k': [10, 100, 500]}
svcp = Pipeline([
  ('feature_selection', NaiveFeatureSelection()),
  ('classification', LinearSVC())
])
clf = GridSearchCV(svcp, parameters, cv=5)
clf.fit(X_train, y_train)
clf.best_params_
print("Best cross validated k:\t%0.0f" % clf.best_params_['feature_selection__k'])

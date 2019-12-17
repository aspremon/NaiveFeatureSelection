# %% Import packages
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from naive_feature_selection import NaiveFeatureSelection
from sklearn.pipeline import Pipeline

# %% Get Breast cancer data set and binarize dataset based on
# median value of features

print("Testing BNFS ...")
print("Loading Breast cancer dataset:")
data = load_breast_cancer(return_X_y=False)
X = 1.0 * np.array([x_ > np.median(x_) for x_ in data['data'].T]).T

X_train, X_test, y_train, y_test = train_test_split(X, data['target'],
                                    test_size = 0.7, random_state=42)


# %% Test Naive Feature Selection, followed by l2 SVM
kv = 15  # kv is target number of features
nfs = NaiveFeatureSelection(k=kv, alpha=1e-3)
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
print('Selected features:')
print([i for i,x in enumerate(nfs.mask_) if x])

# %% Cross validate to get best k
from sklearn.model_selection import GridSearchCV
parameters = {'feature_selection__k': [5,10,15,20,25,30],
   'feature_selection__alpha': [1e-10,1e-5,1e-3,1e-2,1e-1]}
svcp = Pipeline([
  ('feature_selection', NaiveFeatureSelection()),
  ('classification', LinearSVC())
])
clf = GridSearchCV(svcp, parameters, cv=5)
clf.fit(X_train, y_train)
clf.best_params_
print("Best cross validated k: " + str(clf.best_params_['feature_selection__k']))
print("Best cross validated alpha: " + str(clf.best_params_['feature_selection__alpha']))


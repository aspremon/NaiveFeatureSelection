NFS: Naive Feature Selection
=======

This package solves the Naive Feature Selection problem described in [the paper](https://arxiv.org/abs/1905.09884).

# Installation

## Get code and dependencies

- `git clone https://github.com/aspremon/NaiveFeatureSelection`
- `cd NaiveFeatureSelection`
- Install the dependencies listed in `environment.yml`
  - In an existing conda environment, `conda env update -f environment.yml`
  - In a new environment, `conda env create -f environment.yml`, will create a conda environment named `NaiveFeatureSelection`

To check that everything is going well, run `python examples/demoNFS.py`, which should run an example on *20newsgroup* using *scikit-learn*. 

## Install `NaiveFeatureSelection` package

To be able to import and use `NaiveFeatureSelection` in another project, go to your `NaiveFeatureSelection` folder and run `pip install .`

# Usage 

## Minimal usage script

See [demoNFS.py](demoNFS.py)

This script loads the *20newsgroup* text data set from *scikit-learn* and compares accuracy of Naive Feature Selection (followed by SVM), with that of Recursive Feature Elimination (select features by SVM, and run SVM a second time).

```python
from NFS import *
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics


# Get 20newsgroup data set, cf. "Classification of text documents using sparse features" in sklearn doc.
print("Testing NFS ...")
categories = [
        'sci.med',
        'sci.space'
    ]
remove = ('headers', 'footers', 'quotes')
print("Loading 20 newsgroups dataset for categories:")
print(categories if categories else "all")
print()
data_train = fetch_20newsgroups(subset='train', categories=categories,shuffle=True, random_state=42,remove=remove)
data_test = fetch_20newsgroups(subset='test', categories=categories,shuffle=True, random_state=42,remove=remove)


# split training / test set
y_train, y_test = data_train.target, data_test.target


# Vectorize data set
print("Extracting features from the training data using a sparse vectorizer")
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
X_test = vectorizer.transform(data_test.data)
print("n_samples: %d, n_features: %d" % X_test.shape)
print()
feature_names = vectorizer.get_feature_names()


# Test Naive Feature Selection, followed by l2 SVM
k=100 # Target number of features
nfs_res=nfs(X_train,y_train,k)
clfsv = LinearSVC(random_state=0, tol=1e-5)
clfsv.fit(X_train[:,nfs_res['idx']], y_train==1)
y_pred_NFS = clfsv.predict(X_test[:,nfs_res['idx']])
score_nfs = metrics.accuracy_score(y_test==1, y_pred_NFS)
print("NFS accuracy:\t%0.3f" % score_nfs)


# Test Simple Recursive Feature Elimination using SVM
clfrfe = LinearSVC(random_state=0, tol=1e-5)
clfrfe.fit(X_train, y_train==1)
idx=np.argsort(np.squeeze(np.abs(clfrfe.coef_)))[-k:]
clfrfe = LinearSVC(random_state=0, tol=1e-5)
clfrfe.fit(X_train[:,idx], y_train==1)
y_pred_rfe = clfsv.predict(X_test[:,idx])
score_rfe = metrics.accuracy_score(y_test==1, y_pred_rfe)
print("RFE accuracy:\t%0.3f" % score_rfe)
print()


# List selected features
print('Space features:')
print([feature_names[nfs_res['idx'][i]] for i in range(100) if clfsv.coef_[0][i]>=0])
print()
print('Med features:')
print([feature_names[nfs_res['idx'][i]] for i in range(100) if clfsv.coef_[0][i]<0])

```

This should produce the following output.

```
Testing NFS ...

Loading 20 newsgroups dataset for categories:
['sci.med', 'sci.space']

Extracting features from the training data using a sparse vectorizer
n_samples: 1187, n_features: 21368

Extracting features from the test data using the same vectorizer
n_samples: 790, n_features: 21368

NFS accuracy: 0.843
RFE accuracy: 0.592

Space features:
['commercial', 'launches', 'project', 'launched', 'data', 'dryden', 'mining', 'planetary', 'proton', 'missions', 'cost', 'command', 'comet', 'jupiter', 'apollo', 'russian', 'aerospace', 'sun', 'mary', 'payload', 'gravity', 'pat', 'satellites', 'software', 'centaur', 'astronomy', 'landing', 'shafer', 'built', 'titan', 'program', 'vehicle', 'ssto', 'funding', 'flight', 'orbit', 'orbital', 'nasa', 'government', 'moon', 'mission', 'earth', 'allen', 'dc', 'ames', 'rocket', 'rockets', 'satellite', 'mars', 'lunar', 'shuttle', 'solar', 'billion', 'space', 'spacecraft', 'station', 'launch']

Med features:
['med', 'yeast', 'diseases', 'allergic', 'doctors', 'symptoms', 'syndrome', 'diagnosed', 'health', 'drugs', 'therapy', 'candida', 'seizures', 'lyme', 'food', 'brain', 'foods', 'geb', 'pain', 'gordon', 'patient', 'n3jxp', 'patients', 'msg', 'pitt', 'dsl', 'drug', 'doctor', 'disease', 'diet', 'surrender', 'medicine', 'medical', 'chastity', 'intellect', 'treatment', 'cancer', 'cadre', 'shameful', 'skepticism', 'blood', 'soon', 'banks']
```



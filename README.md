
NFS: Naive Feature Selection
=======

# Overview
This package solves the Naive Feature Selection problem described in [the paper](https://arxiv.org/abs/1905.09884). The aforementioned paper introduces sparsity to the Naive Bayes classifier for binary classification. The algorithm in the paper assumes the data is binary (a data matrix with entries in {0,1} commonly referred to as Bernoulli Naive Bayes) or the data is integer-valued and positive (commonly referred to as Multinomial Naive Bayes). 

``` DemoBNFS.py ``` is sample code for the sparse bernoulli naive bayes feature selection.
``` DemoNFS.py ``` is sample code for the sparse multinomial naive bayes feature selection.

Both scripts above employ a 2-stage procedure whereby the naive feature selection algorithm is used to select the important features and then another method (such as SVM, or logistic regression) is used for classification using the features learned by the naive feature selection algorithm (see section 4.2 in paper for more details).

# Installation

```
pip install git+https://github.com/aspremon/NaiveFeatureSelection
```

# Usage

## Hyperparameters

There are two hyperparameters that can be tuned for the naive feature selection algorithm. They are
* k: desired sparsity level
* alpha: laplace smoothing hyper-parameter

Depending on the application, the desired sparsity level may already be known beforehand (i.e., you know the total number of features you want to select). However, alpha remains a hyperparameter and has the same effect as the same hyperparameter for scikit-learn's multinomial naive bayes classifier. We recommend tuning this hyperparameter using cross-validation.

## Minimal usage script

The [DemoBNFS.py](DemoBNFS.py) script loads the *breast cancder* data set from *scikit-learn*, converts it to binary data by thresholding each feature by its median value, and reports accuracy of Naive Feature Selection, followed by SVC using the selected features. The [DemoNFS.py](DemoNFS.py) script loads the *20 newsgroups* text data set from *scikit-learn* and reports accuracy of Naive Feature Selection, followed by SVC using the selected features. 

The package is compatible with *scikit-learn*'s *Fit-Transform* paradigm. To demonstrate this, [DemoNFS.py](DemoNFS.py) runs the same test using the *pipeline* package from *scikit-learn* and performs cross validation using *GridSearchCV* from *sklearn.model_selection*.

To run the `DemoNFS.py` script, type
```
python DemoNFS.py
```

This should produce the following output

```
Testing NFS ...
Loading 20 newsgroups dataset for categories:
['sci.med', 'sci.space']

Extracting features from the training data using a sparse vectorizer
n_samples: 1187, n_features: 21368

Extracting features from the test data using the same vectorizer
n_samples: 790, n_features: 21368

NFS accuracy:   0.843

Space features:
['aerospace', 'allen', 'ames', 'apollo', 'astronomy', 'billion', 'built', 'centaur', 'comet', 'command', 'commercial', 'cost', 'data', 'dc', 'dryden', 'earth', 'flight', 'funding', 'government', 'gravity', 'jupiter', 'landing', 'launch', 'launched', 'launches', 'lunar', 'mars', 'mary', 'mining', 'mission', 'missions', 'moon', 'nasa', 'orbit', 'orbital', 'pat', 'payload', 'planetary', 'program', 'project', 'proton', 'rocket', 'rockets', 'russian', 'satellite', 'satellites', 'shafer', 'shuttle', 'software', 'solar', 'space', 'spacecraft', 'ssto', 'station', 'sun', 'titan', 'vehicle']

Med features:
['allergic', 'banks', 'blood', 'brain', 'cadre', 'cancer', 'candida', 'chastity', 'diagnosed', 'diet', 'disease', 'diseases', 'doctor', 'doctors', 'drug', 'drugs', 'dsl', 'food', 'foods', 'geb', 'gordon', 'health', 'intellect', 'lyme', 'med', 'medical', 'medicine', 'msg', 'n3jxp', 'pain', 'patient', 'patients', 'pitt', 'seizures', 'shameful', 'skepticism', 'soon', 'surrender', 'symptoms', 'syndrome', 'therapy', 'treatment', 'yeast']

Pipeline accuracy:      0.843

Best cross validated k: 500
```


NFS: Naive Feature Selection
=======

This package solves the Naive Feature Selection problem described in [the paper](https://arxiv.org/abs/1905.09884).

# Installation

```
pip install git+https://github.com/aspremon/NaiveFeatureSelection
```

# Usage 

## Minimal usage script

The [DemoNFS.py](DemoNFS.py) script loads the *20 newsgroups* text data set from
*scikit-learn* and compares accuracy of Naive Feature Selection (followed by
SVM), with that of Recursive Feature Elimination (select features by SVM, and
run SVM a second time).

To run the `DemoNFS.py` script:
```
python DemoNFS.py
```

which should produce the following output:


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

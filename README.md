
NFS: Naive Feature Selection
=======

This package solves the Naive Feature Selection problem described in [the paper](https://arxiv.org/abs/1905.09884).

# Installation

```
pip install git+https://github.com/aspremon/NaiveFeatureSelection
```

# Usage

## Minimal usage script

The [DemoNFS.py](DemoNFS.py) script loads the *20 newsgroups* text data set from *scikit-learn* and reports accuracy of Naive Feature Selection, followed by SVC using the selected features.

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


from NFS import *
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics


# Get 20newsgroup data set, cf. "Classification of text documents using sparse features" in sklearn doc.
print("Testing NFS ...")
categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
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
clfsv.fit(X_train[:,nfs_res['idx']], y_train==3)
y_pred_NFS = clfsv.predict(X_test[:,nfs_res['idx']])
score_nfs = metrics.accuracy_score(y_test==3, y_pred_NFS)
print("NFS accuracy:\t%0.3f" % score_nfs)


# Test Simple Recursive Feature Elimination using SVM
clfrfe = LinearSVC(random_state=0, tol=1e-5)
clfrfe.fit(X_train, y_train==3)
idx=np.argpartition(np.squeeze(np.abs(clfrfe.coef_)), -k)[-k:]
clfrfe = LinearSVC(random_state=0, tol=1e-5)
clfrfe.fit(X_train[:,idx], y_train==3)
y_pred_rfe = clfsv.predict(X_test[:,idx])
score_rfe = metrics.accuracy_score(y_test==3, y_pred_rfe)
print("RFE accuracy:\t%0.3f" % score_rfe)
print()

# List selected features
print([feature_names[i] for i in nfs_res['idx']])

import pandas as pd
import numpy as np
from collections import defaultdict
import string
import matplotlib.pyplot as plt
from sklearn import svm
import time

def accuracy(a,b):
    l = len(a)
    if l != len(b):
        print("error: length doesn't comply!")
        return
    cor = 0.0
    for i,j in zip(a,b):
        if i == j:
            cor += 1
    return cor/l

def tf_feature(datum):
    feat = [0]*len(words)
    r = ''.join([c for c in datum['reviews'].lower() if not c in punctuation])
    for w in r.split():
        if w in words:
            feat[word_id[w]] += 1
    return feat

def tfidf_feature(datum):
    feat = tf_feature(datum)
    for w in words:
        feat[word_id[w]] *= idf[w]
    return feat

#Data prepossessing
df = pd.read_csv("new_shuffle_data.csv", names=['product', 'brand', 'price', 'rating', 'reviews', 're_votes'])
df.fillna('', inplace=True)
df = df[1:]
data = df.T.to_dict().values()
data_train = data[:50000]
data_test = data[50000:]

unigram = defaultdict(int)
idf = defaultdict(float)
bigram = defaultdict(int)
bi_idf = defaultdict(float)
trigram = defaultdict(int)
df = pd.read_csv("unigram_tf_idf.csv", names=['word', 'tf', 'idf'])
temp = df[1:].T.to_dict().values()
for d in temp:
    unigram[d['word']] = int(d['tf'])
    idf[d['word']] = float(d['idf'])
df = pd.read_csv("filtered_bigram_tf_idf.csv",names=['bi', 'word1', 'word2', 'tf', 'idf'])
temp = df[1:].T.to_dict().values()
for d in temp:
    bigram[d['word1'], d['word2']] = int(d['tf'])
    bi_idf[d['word1'], d['word2']] = float(d['idf'])
df = pd.read_csv("filtered_trigram_tf.csv", names=['tri', 'word1', 'word2', 'word3', 'tf'])
temp = df[1:].T.to_dict().values()
for d in temp:
    trigram[d['word1'], d['word2'], d['word3']] = int(d['tf'])

punctuation = set(string.punctuation)
words = unigram.keys()[:1000]
word_id = dict(zip(words, range(len(words))))

X_train = [tfidf_feature(d) for d in data_train]
y_train = [d['rating'] for d in data_train]
X_test = [tfidf_feature(d) for d in data_test]
y_test = [d['rating'] for d in data_test]

#Exploratory data analysis
punctuation = set(string.punctuation)
X = [0]*6
Y = [0]*6
Z= [0]*6
for datum in data:
    r = ''.join([c for c in datum['reviews'].lower() if not c in punctuation])
    r = r.split()
    x = int(datum['rating'])
    if 'great' in r:
        X[x] += 1
    if 'works' in r:
        Y[x] += 1
    if 'never' in r:
        Z[x] += 1

L = [1,2,3,4,5]
M = X[1:]
K = Y[1:]
O = Z[1:]

fig,ax = plt.subplots()
bar_width = 0.2
opacity = 0.4
rects1 = plt.bar(L, M, bar_width, alpha = opacity, color='b', label = "#reviews containing 'good'")
rects2 = plt.bar([l+bar_width for l in L], K, bar_width, alpha = opacity, color='r',label = "#reviews containing 'works'")
rects2 = plt.bar([l+2*bar_width for l in L], O, bar_width, alpha = opacity, color='y',label = "#reviews containing 'never'")
plt.xlabel('Rating')
plt.ylabel('number of reviews')
plt.xticks([l+bar_width for l in L], ('1','2','3','4','5'))
plt.ylim(0,12000)
plt.legend()
plt.tight_layout()
plt.show()

#Text model
# model: tfidf + single svm("rbf")
# Train
print "Train model: tf_idf + svm(rbf)"
cls1 = svm.SVC(tol=1e-3, max_iter=1000)
t0 = time.time()
cls1.fit(X_train, y_train)
t1 = time.time()
print "training time: " + str(t1 - t0)

vector1 = cls1.support_vectors_
preds = cls1.predict(X_train)

t2 = time.time()
print "prediction time: " + str(t2 - t1)
acc = accuracy(preds, y_train)
print "Training accuracy = " + str(acc)

# Test
print "Test model: tf_idf + svm(rbf)"
preds = cls1.predict(X_test)
acc = accuracy(preds, y_test)
print "Test accuracy = " + str(acc)

#model: tf_idf + random_forest
from sklearn.ensemble import RandomForestClassifier
f = open("training_test_accuracy.txt","a")

for leaf in [2]:
    for n_estimators in [100, 300, 600]:
        print("Train model: tf_idf +  random_forest, n_estimators = {0}, min_samples_leaf = {1}\n".format(n_estimators, leaf))
        f.write("Train model: tf_idf +  random_forest, n_estimators = {0}, min_samples_leaf = {1}\n".format(n_estimators, leaf))
        #Train
        t0 = time.time()
        cls2 = RandomForestClassifier(n_estimators=n_estimators,
                                      min_samples_leaf=leaf,
                                      class_weight = "balanced",
                                      n_jobs=15)
        cls2.fit(X_train, y_train)
        t1 = time.time()
        print("Prediction time: " + str(t1-t0))
        acc = cls2.score(X_train,y_train)
        print("Training accuracy = " + str(acc))
        f.write("Training accuracy = {}".format(acc))

        #Test
        acc = cls2.score(X_test, y_test)
        print("Test accuracy = " + str(acc))
        f.write("Test accuracy = {}".format(acc))
f.close()

#model: tf_idf + random_forest
from sklearn.ensemble import RandomForestClassifier
f = open("training_test_accuracy.txt","a")

for leaf in [2]:
    for n_estimators in [100, 300, 600]:
        print("Train model: tf_idf +  random_forest, n_estimators = {0}, min_samples_leaf = {1}\n".format(n_estimators, leaf))
        f.write("Train model: tf_idf +  random_forest, n_estimators = {0}, min_samples_leaf = {1}\n".format(n_estimators, leaf))
        #Train
        t0 = time.time()
        cls2 = RandomForestClassifier(n_estimators=n_estimators,
                                      min_samples_leaf=leaf,
                                      class_weight = "balanced",
                                      n_jobs=15)
        cls2.fit(X_train, y_train)
        t1 = time.time()
        print("Prediction time: " + str(t1-t0))
        acc = cls2.score(X_train,y_train)
        print("Training accuracy = " + str(acc))
        f.write("Training accuracy = {}".format(acc))

        #Test
        acc = cls2.score(X_test, y_test)
        print("Test accuracy = " + str(acc))
        f.write("Test accuracy = {}".format(acc))
f.close()

# model: tfidf + Bagging svm("rbf")
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier

f = open("training_test_accuracy.txt", "a")

for kernel in ['sigmoid', 'rbf']:
    print("Train model: tf_idf + Bagging SVC('{}')".format(kernel))
    f.write("Train model: tf_idf + Bagging SVC('{}')".format(kernel))
    # Train
    n_estimators = 10
    t0 = time.time()
    cls3 = OneVsRestClassifier(BaggingClassifier(svm.SVC(kernel=kernel),
                                                 max_samples=1.0 / n_estimators,
                                                 n_estimators=n_estimators,
                                                 n_jobs=15))
    cls3.fit(X_train, y_train)
    t1 = time.time()
    print("Prediction time: " + str(t1 - t0))
    acc = cls3.score(X_train, y_train)
    print("Training accuracy = " + str(acc))
    f.write("Training accuracy = {}".format(acc))

    # Test
    acc = cls3.score(X_test, y_test)
    print("Test accuracy = " + str(acc))
    f.write("Test accuracy = {}".format(acc))
f.close()

#model: tfidf + Linearsvm()
from sklearn import svm
import time

f = open("training_test_accuracy.txt","a")

#Train balanced class weight
print "Train model: tf_idf +  Linearsvm(), class_weight = 'balanced'"
f.write("Train model: tf_idf +  Linearsvm(), class_weight = 'balanced'\n")
cls4 = svm.LinearSVC(max_iter=1000, tol=1e-5, class_weight = 'balanced')
t0 = time.time()
cls4.fit(X_train, y_train)
t1 = time.time()
print "training time: " + str(t1-t0)

preds = cls4.predict(X_train)
t2 = time.time()
print "prediction time: " + str(t2-t1)
acc = accuracy(preds, y_train)
print "Training accuracy = " + str(acc)
f.write("Training accuracy = {}\n".format(acc))
#Test
print "Test model: tf_idf +  Linearsvm()"
preds = cls4.predict(X_test)
acc = accuracy(preds, y_test)
print "Test accuracy = " + str(acc)
f.write("Test accuracy = {}\n".format(acc))
f.write("\n")

#Train without balance weight
print "Train model: tf_idf +  Linearsvm(), class_weight = 'not balanced'"
f.write("Train model: tf_idf +  Linearsvm(), class_weight = 'not balanced'\n")
cls4 = svm.LinearSVC(max_iter=1000, tol=1e-5)
t0 = time.time()
cls4.fit(X_train, y_train)
t1 = time.time()
print "training time: " + str(t1-t0)

preds = cls4.predict(X_train)
t2 = time.time()
print "prediction time: " + str(t2-t1)
acc = accuracy(preds, y_train)
print "Training accuracy = " + str(acc)
f.write("Training accuracy = {}\n".format(acc))
#Test
print "Test model: tf_idf +  Linearsvm()"
preds = cls4.predict(X_test)
acc = accuracy(preds, y_test)
print "Test accuracy = " + str(acc)
f.write("Test accuracy = {}\n".format(acc))
f.write("\n")
f.close()

# model: tf + LSA + KNN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
import time

# tf formalization
n_features = 10000
review_train = [d["reviews"] for d in data_train]
review_test = [d["reviews"] for d in data_test]
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=0.,
                                max_features=n_features,
                                strip_accents='ascii',
                                decode_error='ignore',
                                stop_words='english')
t0 = time.time()
X_train = tf_vectorizer.fit_transform(review_train)
X_test = tf_vectorizer.fit_transform(review_test)
y_train = [d['rating'] for d in data_train]
y_test = [d['rating'] for d in data_test]
t1 = time.time()
print "tf transformation time: " + str(t1 - t0)

# Training
f = open("training_test_accuracy.txt", "a")
for n_components in [500, 1000, 1500, 2000]:
    svd = TruncatedSVD(n_components=n_components, n_iter=10, random_state=42)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    t0 = time.time()
    lsa_train = lsa.fit_transform(X_train)
    lsa_test = lsa.fit_transform(X_test)

    print "Train model: tf + LSA + KNN, n_features=10000, n_components={}".format(n_components)
    f.write("Train model: tf + LSA + KNN, n_features=10000, n_components={}\n".format(n_components))
    knn_lsa = KNeighborsClassifier(n_neighbors=10000, algorithm='auto', metric='cosine')
    knn_lsa.fit(lsa_train, y_train)
    t1 = time.time()
    print "Training time = " + str(t1 - t0)

    t2 = time.time()
    preds = knn_lsa.predict(lsa_train)
    acc = accuracy(preds, y_train)
    print "Prediction time = " + str(t2 - t1)
    print "Training accuracy = " + str(acc)
    f.write("Training accuracy = {}\n".format(acc))

    # Test
    print "Test model: tfidf + LSA + KNN"
    preds = knn_lsa.predict(lsa_test)
    acc = accuracy(preds, y_test)
    print "Test accuracy = " + str(acc)
    f.write("Test accuracy = {}\n".format(acc))
    f.write("\n")
f.close()

# model: tfidf + LSA + svm('rbf')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import svm
import time

# tfidf formalization
n_features = 10000
review_train = [d["reviews"] for d in data_train]
review_test = [d["reviews"] for d in data_test]
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.,
                                   max_features=n_features,
                                   strip_accents='unicode',
                                   decode_error='ignore',
                                   stop_words='english')
t0 = time.time()
X_train = tfidf_vectorizer.fit_transform(review_train)
X_test = tfidf_vectorizer.fit_transform(review_test)
y_train = [d['rating'] for d in data_train]
y_test = [d['rating'] for d in data_test]
t1 = time.time()

# Training
f = open("training_test_accuracy.txt", "a")
for n_components in [500, 1000]:
    svd = TruncatedSVD(n_components=n_components, n_iter=10, random_state=42)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    t0 = time.time()
    lsa_train = lsa.fit_transform(X_train)
    lsa_test = lsa.fit_transform(X_test)

    print "Train model: tf + LSA + svm(), n_features=10000, n_components={}".format(n_components)
    f.write("Train model: tf + LSA + svm(), n_features=10000, n_components={}\n".format(n_components))
    svm_lsa = svm.SVC(tol=1e-3, max_iter=1000)
    svm_lsa.fit(lsa_train, y_train)
    t1 = time.time()
    print "Training time = " + str(t1 - t0)

    t2 = time.time()
    preds = svm_lsa.predict(lsa_train)
    acc = accuracy(preds, y_train)
    print "Prediction time = " + str(t2 - t1)
    print "Training accuracy = " + str(acc)
    f.write("Training accuracy = {}\n".format(acc))

    # Test
    print "Test model: tfidf + LSA + KNN"
    preds = svm_lsa.predict(lsa_test)
    acc = accuracy(preds, y_test)
    print "Test accuracy = " + str(acc)
    f.write("Test accuracy = {}\n".format(acc))
    f.write("\n")
f.close()
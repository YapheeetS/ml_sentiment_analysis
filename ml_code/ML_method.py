from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import tensorflow as tf
import os
import sklearn.metrics as metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from tqdm import tqdm

MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 1000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

x_train = np.load('x_train.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)
x_test = np.load('x_test.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)

vocabulary = []
with open('./aclImdb/imdb.vocab') as f:
    vocabulary = f.read().splitlines()


for i, d in tqdm(enumerate(x_train)):
    sentence = ' '.join(x_train[i])
    x_train[i] = sentence

for i, d in tqdm(enumerate(x_test)):
    sentence = ' '.join(x_test[i])
    x_test[i] = sentence

# count_vect = CountVectorizer(min_df=0.001, max_df=0.5, max_features=400)
count_vect = CountVectorizer(vocabulary=vocabulary)
X_train_counts = count_vect.fit_transform(x_train)
X_test_counts = count_vect.fit_transform(x_test)

count_vect = TfidfTransformer()
tf_transformer = TfidfTransformer().fit(X_train_counts)
x_train = tf_transformer.transform(X_train_counts)
x_train = x_train.toarray()
print(x_train.shape)
    

tf_transformer = TfidfTransformer().fit(X_test_counts)
x_test = tf_transformer.transform(X_test_counts)


def lg_fit_and_predicted(train_x, train_y, test_x, test_y, penalty='l2', C=1.0, solver='lbfgs'):

    print('training ...')
    clf = LogisticRegression(penalty=penalty, C=C, solver=solver, n_jobs=1).fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    print('...LogisticRegression...')
    print(metrics.classification_report(test_y, predict_y, digits=5))


def svm_fit_and_predicted(train_x, train_y, test_x, test_y, C=1.0):
    print('training ...')
    clf = LinearSVC(C=1.0).fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    print('...SVM...')
    print(metrics.classification_report(test_y, predict_y, digits=5))


def bayes_fit_and_predicted(train_x, train_y, test_x, test_y, C=1.0):
    print('training ...')
    clf = MultinomialNB().fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    print('...Bayes...')
    print(metrics.classification_report(test_y, predict_y, digits=5))


lg_fit_and_predicted(x_train, y_train, x_test, y_test)
svm_fit_and_predicted(x_train, y_train, x_test, y_test)
bayes_fit_and_predicted(x_train, y_train, x_test, y_test)

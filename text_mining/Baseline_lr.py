import csv
from collections import defaultdict
import random
import numpy
import sklearn
from sklearn import linear_model
import scipy.optimize
import pandas
import math
def feature(datum):
    feat_color = [0] * len(colorDict)
    feat_storage = [0] * len(storageDict)
    feat_carrier = [0] * len(carrierDict)
    feat_warranty = [0] * len(warrantyDict)
    feat_unlock = [0] * len(unlockDict)
    feat_phonetype = [0] * len(phonetypeDict)

    feat_color[int(datum[0])] = 1
    feat_storage[int(datum[1])] = 1
    feat_carrier[int(datum[2])] = 1
    feat_warranty[int(datum[3])] = 1
    feat_unlock[int(datum[4])] = 1
    feat_phonetype[int(datum[5])] = 1
    feat_price = [float(datum[6])]

    feat = [1] + feat_color + feat_storage + feat_carrier + feat_warranty + \
           feat_unlock + feat_phonetype + feat_price

    return feat

def getAcc(predictions, y):
    acc = sum([y[i] == predictions[i] for i in xrange(len(y))])
    acc /= 1.0 * len(y)
    print acc

def regression(X, y, X_test):
    clf = linear_model.Ridge(1.0, fit_intercept=False)
    clf.fit(X, y)
    theta = clf.coef_
    predictions = clf.predict(X_test)
    print theta
    return predictions

def approx(predictions):
    predictions = [int(round(i)) for i in predictions]

    for i in xrange(len(predictions)):
        if predictions[i] < 1:
            predictions[i] = 1
        elif predictions[i] > 5:
            predictions[i] = 5

    return predictions

def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= math.log(1 + math.exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  # for debugging
  print("ll = " + str(loglikelihood))
  return -loglikelihood

def fprime(theta, X, y, lam):
  dl = [0]*len(theta)
  for i in range(len(X)):
    logit = inner(X[i], theta)
    for k in range(len(theta)):
      dl[k] += X[i][k] * (1 - sigmoid(logit))
      if not y[i]:
        dl[k] -= X[i][k]
  for k in range(len(theta)):
    dl[k] -= lam*2*theta[k]
  return numpy.array([-x for x in dl])

def inner(x, y):
  sum = 0
  for a,b in zip(x,y):
    sum += a*b
  return sum

def sigmoid(x):
  return 1.0 / (1 + math.exp(-x))

def regression2(X, y, X_test):
    lam = 1.0
    theta,_,_ = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, pgtol = 1, args = (X, y, lam))
    predictions = [inner(x,theta) for x in X_test]
    print theta
    return predictions

#Data prepossessing
data = []
with open("new_shuffle_data_without_NA.csv",'rb') as my_file:
    reader = csv.reader(my_file)
    data = list(reader)
priceList = [d[2] for d in data]
len(priceList)

propertyList = []
with open("property_feature_update2.csv", 'r') as my_file:
    reader = csv.reader(my_file)
    propertyList = list(reader)
print len(propertyList)
print propertyList[:10]

colorSet = set([f[0] for f in propertyList[1:]])
colorDict = {v: k for k, v in dict(enumerate(sorted(list(colorSet)))).iteritems()}

storageSet = set([int(f[1]) for f in propertyList[1:]])
storageDict = {str(v): k for k, v in dict(enumerate(sorted(list(storageSet)))).iteritems()}

carrierSet = set([f[2] for f in propertyList[1:]])
carrierDict = {v: k for k, v in dict(enumerate(sorted(list(carrierSet)))).iteritems()}

phonetypeSet = set([f[5] for f in propertyList[1:]])
phonetypeDict = {v: k for k, v in dict(enumerate(sorted(list(phonetypeSet)))).iteritems()}

warrantyDict = {'0': 0, '1': 1, '2': 2}
unlockDict = {'0': 0, '1': 1}

with open('property_feature_encoded2.csv', 'wb') as out_file:
    writer = csv.writer(out_file,delimiter = ',')
    writer.writerow(propertyList[0] + ['price'])
    count = 1
    for r in propertyList[1:]:
        d = []
        d.append(colorDict[str(r[0])])
        d.append(storageDict[str(r[1])])
        d.append(carrierDict[str(r[2])])
        d.append(warrantyDict[str(r[3])])
        d.append(unlockDict[str(r[4])])
        d.append(phonetypeDict[str(r[5])])
        d.append(priceList[count])
        count += 1
        writer.writerow(d)

features = []
with open("property_feature_encoded2.csv",'rb') as my_file:
    reader = csv.reader(my_file)
    features = list(reader)

data_train = data[1:50001]
data_test = data[50001:]
features_train = features[1:50001]
features_test = features[50001:]

X_train = [feature(d) for d in features_train]
y_train = [int(d[3]) for d in data_train]
X_test = [feature(d) for d in features_test]
y_test = [int(d[3]) for d in data_test]

#Traning
predictions =  approx(regression2(X_train, y_train, X_test))
#Test
getAcc(predictions, y_test)
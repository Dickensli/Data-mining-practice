### Would-visit baseline: just rank which businesses are popular and which are not, and return '1' if a business is among the top-ranked
from util import *
from collections import defaultdict
import random

users = []
busi = []
pairs = defaultdict(list)
val_gt = list()
user_cat = defaultdict(set)
busi_cat = defaultdict(set)

def jaccard(it1, it2):
    intsec = float(len(set(it1).intersection(it2)))
    uni = float(len(set(it1).union(it2)))
    return intsec/uni

def prediction(user, business):
    for b in u_pairs[user]:
        if jaccard(i_pairs[business], i_pairs[b]) > 0:
            return 1
    return 0

#data prepossessing
users = set()
busi = set()
visitedSet = set()
notVisited = set()
u_pairs = defaultdict(list)
i_pairs = defaultdict(list)
val_gt = list()

i = 0
for l in readGz("train.json.gz"):
    user, business = l['userID'], l['businessID']
    users.add(user)
    busi.add(business)
    visitedSet.add((user, business))
    if i < 100000:
        u_pairs[user].append(business)
        i_pairs[business].append(user)
        i += 1
    else:
        val_gt.append((user, business, 1))

#Collaborative filter
user_list = list(users)
busi_list = list(busi)
while (len(notVisited) < 100000):
    u = random.choice(user_list)
    b = random.choice(busi_list)
    if (u, b) in visitedSet or (u, b) in notVisited: continue
    notVisited.add((u, b))
    val_gt.append((u, b, 0))

#Predict
correct = 0
for val in val_gt:
    pred = prediction(val[0], val[1])
    correct += 1 if pred == val[2] else 0
acc = float(correct) / 200000

#Store the result
predictions = open("predictions_Visit.txt", 'w')
for l in open("pairs_Visit.txt"):
    if l.startswith("userID"):
        #header
        predictions.write(l)
        continue
    u,i = l.strip().split('-')
    pred = prediction(u, i)
    predictions.write(u + '-' + i + "," + str(pred) + "\n")

predictions.close()
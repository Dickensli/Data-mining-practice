### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before
import numpy as np
import random
from sklearn.metrics import mean_squared_error

#Data prepossessing
u_pairs = {}
i_pairs = {}
u_i_r = {}
allRatings = []
val_gt = []
i = 0
for l in readGz("train.json.gz"):
    user,business,rate = l['userID'],l['businessID'],l['rating']
    if i < 200000:
        allRatings.append(rate)
        u_pairs[user] = u_pairs.get(user,[]) + [business]
        i_pairs[business] = i_pairs.get(business,[]) + [user]
        u_i_r[(user,business)] = rate
        i += 1
    else:
        val_gt.append((user,business,rate))

globalAverage = sum(allRatings) / len(allRatings)

#Objective function
def obj(lam1, lam2, lam3, alpha, beta_u, beta_i, gamma_u, gamma_i):
    f = g1 = g2 = g3 = 0
    for user,business in u_i_r.keys():
        f += (alpha + beta_u[user] + beta_i[business] + gamma_u[user].dot(gamma_i[business]) - u_i_r[(user,business)])**2
    for user in u_pairs.keys():
        g1 += beta_u[user]**2
        for k in range(1):
            g3 += gamma_u[user][k]**2
    for business in i_pairs.keys():
        g2 += beta_i[business]**2
        for k in range(1):
            g3 += gamma_i[business][k]**2
    return f + lam1*g1 + lam2*g2 + lam3*g3

#Train
def li_train(lam1, lam2, lam3):
    alpha = globalAverage
    beta_u = {}
    beta_i = {}
    K = 1
    gamma_u = defaultdict(list)
    gamma_i = defaultdict(list)

    # initialization
    for user in u_pairs.keys():
        beta_u[user] = 0
        gamma_u[user] = np.array([0] * K)
    for business in i_pairs.keys():
        beta_i[business] = 0
        gamma_i[business] = np.array([0] * K)

    for i in range(40):
        # alpha
        alpha_nom = 0
        for user, business in u_i_r.keys():
            alpha_nom += u_i_r[(user, business)] - beta_u[user] - beta_i[business] - gamma_u[user].dot(
                gamma_i[business])
        alpha = float(alpha_nom) / len(u_i_r)

        # beta_u
        for user in u_pairs.keys():
            beta_u_nom = 0
            for business in u_pairs[user]:
                beta_u_nom += u_i_r[(user, business)] - alpha - beta_i[business] - gamma_u[user].dot(gamma_i[business])
            beta_u[user] = float(beta_u_nom) / (lam1 + len(u_pairs[user]))

        # beta_i
        for business in i_pairs.keys():
            beta_i_nom = 0
            for user in i_pairs[business]:
                beta_i_nom += u_i_r[(user, business)] - alpha - beta_u[user] - gamma_u[user].dot(gamma_i[business])
            beta_i[business] = float(beta_i_nom) / (lam2 + len(i_pairs[business]))

            # gamma_u
        for user in u_pairs.keys():
            for k in range(K):
                gamma_u_den = float(lam3)
                gamma_u_nom = 0.0
                for business in u_pairs[user]:
                    gamma_u_nom += (u_i_r[(user, business)] - alpha - beta_u[user] - beta_i[business] - \
                                    gamma_u[user].dot(gamma_i[business])) * \
                                   gamma_i[business][k]
                    gamma_u_den += gamma_i[business][k] ** 2
                gamma_u[user][k] = gamma_u_nom / gamma_u_den

                # gamma_i
        for business in i_pairs.keys():
            for k in range(K):
                gamma_i_den = float(lam3)
                gamma_i_nom = 0.0
                for user in i_pairs[business]:
                    gamma_i_nom += (u_i_r[(user, business)] - alpha - beta_u[user] - beta_i[business] - \
                                    gamma_u[user].dot(gamma_i[business])) * \
                                   gamma_u[user][k]
                    gamma_i_den += gamma_u[user][k] ** 2
                gamma_i[business][k] = gamma_i_nom / gamma_i_den

        print(obj(lam1, lam2, lam3, alpha, beta_u, beta_i, gamma_u, gamma_i))

    return alpha, beta_u, beta_i, gamma_u, gamma_i

#Predict
def li_prediction(a, bu, bi, gu, gi, user, business):
    bu_index = bu.get(user, 0)
    bi_index = bi.get(business, 0)
    gu_index = gu.get(user, np.array([0] * 1))
    gi_index = gi.get(business, np.array([0] * 1))
    return a + bu_index + bi_index + gu_index.dot(gi_index)

pred = []
label = []
alpha, beta_u, beta_i, gamma_u, gamma_i = li_train(3.8,5.7,7.0)
for val in val_gt:
    user, business = val[0],val[1]
    label.append(val[2])
    pred.append(li_prediction(alpha, beta_u, beta_i, gamma_u, gamma_i, user, business))
mse = mean_squared_error(pred, label)

#Store the result
predictions = open("predictions_Rating.txt", 'w')
for l in open("pairs_Rating.txt"):
    if l.startswith("userID"):
        #header
        predictions.write(l)
        continue
    u,i = l.strip().split('-')
    pred = li_prediction(alpha, beta_u, beta_i, gamma_u, gamma_i, u, i)
    predictions.write(u + '-' + i + ',' + str(pred) + '\n')
predictions.close()
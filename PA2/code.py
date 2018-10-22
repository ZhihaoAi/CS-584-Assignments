import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB

Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))

train_jll = np.zeros((10, 15))
test_jll = np.zeros((10, 15))

for i_dataset in range(10):
    X, y = Xs[i_dataset], ys[i_dataset]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1./3, random_state=1527)
    y_train_indices = [0 if i==False else 1 for i in y_train]
    y_test_indices = [0 if i==False else 1 for i in y_test]

    for i_alpha in range(-7,8):
        clf = BernoulliNB(alpha=10**i_alpha)
        clf.fit(X_train, y_train)
        sum_train_jll, sum_test_jll = 0, 0
        for i in range(len(y_train)):
            sum_train_jll += clf._joint_log_likelihood(X_train)[i][y_train_indices[i]]    
        for i in range(len(y_test)):
            sum_test_jll += clf._joint_log_likelihood(X_test)[i][y_test_indices[i]]
        train_jll[i_dataset][i_alpha+7] = sum_train_jll
        test_jll[i_dataset][i_alpha+7] = sum_test_jll

pickle.dump((train_jll, test_jll), open('result.pkl', 'wb'))
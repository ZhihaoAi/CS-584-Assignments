import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))

l2_model_complexity = np.zeros((10, 15))
l2_num_zero_weights = np.zeros((10, 15))
l1_num_zero_weights = np.zeros((10, 15))
l2_train_cll = np.zeros((10, 15))
l2_test_cll = np.zeros((10, 15))

def l2_complexity(w0, ws):
    c = w0**2
    for w in ws:
        c += w**2
    return c

def number_of_zeros(w0, ws):
    count = 0
    if w0 == 0:
        count+=1
    count+=ws.tolist().count(0)
    return count

def cll(plp, idx):
    s = 0
    for i in range(len(idx)):
        s += plp[i, idx[i]]
    return s

for i_dataset in range(10):
    X, y = Xs[i_dataset], ys[i_dataset]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1./3, random_state=1527)
    y_train_indices = [0 if i==False else 1 for i in y_train]
    y_test_indices = [0 if i==False else 1 for i in y_test]

    for i_c in range(-7,8):
        clfl2 = LogisticRegression(penalty='l2', C=10**i_c, random_state=42).fit(X_train, y_train)
        l2_model_complexity[i_dataset][i_c+7] = l2_complexity(clfl2.intercept_, clfl2.coef_[0])
        l2_num_zero_weights[i_dataset][i_c+7] = number_of_zeros(clfl2.intercept_, clfl2.coef_[0])
        l2_train_cll[i_dataset][i_c+7] = cll(clfl2.predict_log_proba(X_train), y_train_indices)
        l2_test_cll[i_dataset][i_c+7] = cll(clfl2.predict_log_proba(X_test), y_test_indices)
        
        clfl1 = LogisticRegression(penalty='l1', C=10**i_c, random_state=42).fit(X_train, y_train)
        l1_num_zero_weights[i_dataset][i_c+7] = number_of_zeros(clfl1.intercept_, clfl1.coef_[0])

for i in range(10):
    _, ax = plt.subplots()
    ax.set_title('Dataset %d'%(i+1))
    ax.plot(l2_model_complexity[i], l2_train_cll[i], label='train_cll')
    ax.plot(l2_model_complexity[i], l2_test_cll[i], label='test_cll')
    ax.legend()
plt.show()

for i in range(10):
    _, ax = plt.subplots()
    ax.set_title('Dataset %d'%(i+1))
    ax.plot(np.arange(-7,8), l2_num_zero_weights[i], label='l2_num_zero')
    ax.plot(np.arange(-7,8), l1_num_zero_weights[i], label='l1_num_zero')
    ax.legend()
plt.show()

pickle.dump((l2_model_complexity, l2_train_cll, l2_test_cll, l2_num_zero_weights, l1_num_zero_weights), open('result.pkl', 'wb'))

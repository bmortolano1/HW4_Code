import numpy as np

def calc_stoch_loss_grad(w, C, N, y, x):
    hinge = 1 - y * np.dot(w, x)
    w0 = w[0:-1]

    if hinge > 0: # Incorrect or within margin
        grad = np.append(w0, 0) - C*N*y*x
    else:
        grad = np.append(w0, 0)

    return grad

def stoch_grad_desc(w_in, C, N, y_arr, x_arr, gamma, shuffle_indeces):

    w = w_in
    n_updates = 0

    for i in shuffle_indeces:
        x = x_arr[i]
        x = np.append(x, 1)
        y = y_arr[i]

        w_new = w - gamma * calc_stoch_loss_grad(w, C, N, y, x)

        if all(w != w_new):
            n_updates = n_updates + 1
            w = w_new

    return (w, n_updates)

def train_svm(C, labels, features, gamma0, a, epochs, scheduling):
    rng = np.random.default_rng()
    indeces = np.arange(np.size(labels))
    w = np.zeros(np.size(features,1) + 1)
    N = np.size(labels)
    n_updates = 0

    for T in range(epochs):
        rng.shuffle(indeces)

        if scheduling == 'a':
            gamma = gamma0 / (1+gamma0/a*T)
        elif scheduling == 'b':
            gamma = gamma0 / (1 + T)

        [w, n_updates_epoch] = stoch_grad_desc(w, C, N, labels, features, gamma, indeces)
        n_updates = n_updates + n_updates_epoch
        # print("Weight: " + str(w) + "\nn_updates: " + str(n_updates) + "\n\n")

    return w

def calc_error(w, labels, features):
    n_corr = 0
    n_incorr = 0

    for i in range(np.size(labels)):
        x = np.append(features[i], 1)
        y = labels[i]
        if np.sign(np.dot(w, x)) == np.sign(y):
            n_corr = n_corr + 1
        else:
            n_incorr = n_incorr + 1

    return n_corr / (n_corr + n_incorr)


import scipy.optimize._minimize as minim
import numpy as np

def dual_form_loss(alpha, x, y):
    loss = 0.5 * np.dot(alpha*y, np.dot(np.dot(x, np.transpose(x)), alpha*y)) - np.sum(alpha)

    return loss

def alpha_constraint(alpha, y):
    return np.sum([alpha[i]*y[i] for i in range(np.size(y))])

def optimize_alpha(x_arr, y_arr, C):
    alpha0 = np.zeros(np.size(y_arr))
    constraint = {
        "type" : "eq",
        "fun" : alpha_constraint,
        "args" : (y_arr,)
    }

    bound_tuple = (0,C)
    bounds_ = list((bound_tuple, ) * np.size(y_arr))

    sol = minim.minimize(fun=dual_form_loss, x0=alpha0, args=(x_arr, y_arr), method="SLSQP", bounds=bounds_, constraints=constraint)
    alpha = sol.x

    w = sum(alpha[i]*y_arr[i]*x_arr[i] for i in range(np.size(alpha)) if alpha[i] > 0)
    b = np.average([y_arr[i] - np.dot(w, x_arr[i]) for i in range(np.size(alpha)) if (alpha[i] > 0)])

    return np.concatenate((w,[b]))

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

def gauss_kernel(x1, x2, sigma):
    return np.exp(-np.linalg.norm(x1-x2)**2 / sigma)

def gaussian_kernel_matrix(x_all, sigma):
    d = np.size(x_all,0)
    K = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            K[i,j] = gauss_kernel(x_all[i], x_all[j], sigma)
    return K

def dual_form_loss_gauss_kernel(alpha, x, y, K):
    loss = 0.5 * np.dot(alpha*y, np.dot(K, alpha*y)) - np.sum(alpha)
    return loss

def optimize_alpha_gauss(x_arr, y_arr, C, sigma):
    alpha0 = np.zeros(np.size(y_arr))
    constraint = {
        "type" : "eq",
        "fun" : alpha_constraint,
        "args" : (y_arr,)
    }

    bound_tuple = (0,C)
    bounds_ = list((bound_tuple, ) * np.size(y_arr))

    K = gaussian_kernel_matrix(x_arr, sigma) # Pre-compute this for training features
    sol = minim.minimize(fun=dual_form_loss_gauss_kernel, x0=alpha0, args=(x_arr, y_arr, K), method="SLSQP", bounds=bounds_, constraints=constraint)
    alpha = sol.x

    return alpha

def calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma):
    n_corr = 0
    n_incorr = 0

    # Calculate bias term
    sum = 0
    n = 0
    for i in range(np.size(test_labels)):
        x = train_features[i]
        y = train_labels[i]

        # Calculate prediction using weighted sum of support vectors
        if alpha[i] > 0:
            pred = np.sum([alpha[j]*train_labels[j]*gauss_kernel(train_features[j], x, sigma) for j in range(np.size(train_labels)) if alpha[j] > 0])
            sum = sum + y - pred
            n = n + 1

    bias = sum / n

    for i in range(np.size(test_labels)):
        x_test = test_features[i]
        y_test = test_labels[i]

        # Calculate prediction using weighted sum of support vectors
        pred = np.sum([alpha[j]*train_labels[j]*gauss_kernel(train_features[j], x_test, sigma) for j in range(np.size(train_labels)) if alpha[j] > 0]) + bias

        if np.sign(pred) == np.sign(y_test):
            n_corr = n_corr + 1
        else:
            n_incorr = n_incorr + 1

    return n_corr / (n_corr + n_incorr)
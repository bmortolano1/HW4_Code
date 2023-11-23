import SVM.primal_svm_solver as primal
import SVM.dual_svm_solver as dual
import data_reader as dt
import numpy as np

if __name__ == '__main__':
    [train_features, train_labels] = dt.parse_file("./bank-note/train.csv")
    [test_features, test_labels] = dt.parse_file("./bank-note/test.csv")

    # print("Part 2A")
    #
    # gamma0 = 1
    # a = 0.05
    #
    # w = primal.train_svm(100 / 873, train_labels, train_features, gamma0, a, 100, 'a')
    # print("C: " + str(100 / 873) + ", Train Accuracy: " + str(
    #     primal.calc_error(w, train_labels, train_features)) + ", Test Accuracy: " + str(
    #     primal.calc_error(w, test_labels, test_features)), ", Weights: " + str(w))
    #
    # w = primal.train_svm(500 / 873, train_labels, train_features, gamma0, a, 100, 'a')
    # print("C: " + str(500 / 873) + ", Train Accuracy: " + str(
    #     primal.calc_error(w, train_labels, train_features)) + ", Test Accuracy: " + str(
    #     primal.calc_error(w, test_labels, test_features)), ", Weights: " + str(w))
    #
    # w = primal.train_svm(700 / 873, train_labels, train_features, gamma0, a, 100, 'a')
    # print("C: " + str(700 / 873) + ", Train Accuracy: " + str(
    #     primal.calc_error(w, train_labels, train_features)) + ", Test Accuracy: " + str(
    #     primal.calc_error(w, test_labels, test_features)), ", Weights: " + str(w))
    #
    # print("Part 2B")
    # gamma0 = 1
    #
    # w = primal.train_svm(100 / 873, train_labels, train_features, gamma0, 0, 100, 'b')
    # print("C: " + str(100 / 873) + ", Train Accuracy: " + str(
    #     primal.calc_error(w, train_labels, train_features)) + ", Test Accuracy: " + str(
    #     primal.calc_error(w, test_labels, test_features)), ", Weights: " + str(np.round(w, 3)))
    #
    # w = primal.train_svm(500 / 873, train_labels, train_features, gamma0, 0, 100, 'b')
    # print("C: " + str(500 / 873) + ", Train Accuracy: " + str(
    #     primal.calc_error(w, train_labels, train_features)) + ", Test Accuracy: " + str(
    #     primal.calc_error(w, test_labels, test_features)), ", Weights: " + str(w))
    #
    # w = primal.train_svm(700 / 873, train_labels, train_features, gamma0, 0, 100, 'b')
    # print("C: " + str(700 / 873) + ", Train Accuracy: " + str(
    #     primal.calc_error(w, train_labels, train_features)) + ", Test Accuracy: " + str(
    #     primal.calc_error(w, test_labels, test_features)), ", Weights: " + str(w))

    # print("Part 3A")
    # w = dual.optimize_alpha(train_features, train_labels, 100/873)
    # print("C: " + str(100 / 873) + ", Train Accuracy: " + str(
    #     dual.calc_error(w, train_labels, train_features)) + ", Test Accuracy: " + str(
    #     dual.calc_error(w, test_labels, test_features)), ", Weights: " + str(w))
    #
    # w = dual.optimize_alpha(train_features, train_labels, 500/873)
    # print("C: " + str(500 / 873) + ", Train Accuracy: " + str(
    #     dual.calc_error(w, train_labels, train_features)) + ", Test Accuracy: " + str(
    #     dual.calc_error(w, test_labels, test_features)), ", Weights: " + str(w))
    #
    # w = dual.optimize_alpha(train_features, train_labels, 700/873)
    # print("C: " + str(700 / 873) + ", Train Accuracy: " + str(
    #     dual.calc_error(w, train_labels, train_features)) + ", Test Accuracy: " + str(
    #     dual.calc_error(w, test_labels, test_features)), ", Weights: " + str(w))

    print("Part 3B")

    sigma = 0.1
    alpha = dual.optimize_alpha_gauss(train_features, train_labels, 100/873, sigma)
    print("C: " + str(100 / 873) + ", Sigma: " + str(sigma) + ", Train Accuracy: " + str(
        dual.calc_error_gauss(alpha, train_labels, train_features, train_labels, train_features, sigma)) + ", Test Accuracy: " + str(
        dual.calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma)))
    alpha = dual.optimize_alpha_gauss(train_features, train_labels, 500/873, sigma)
    print("C: " + str(500 / 873) + ", Sigma: " + str(sigma) + ", Train Accuracy: " + str(
        dual.calc_error_gauss(alpha, train_labels, train_features, train_labels, train_features, sigma)) + ", Test Accuracy: " + str(
        dual.calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma)))
    alpha = dual.optimize_alpha_gauss(train_features, train_labels, 700/873, sigma)
    print("C: " + str(700 / 873) + ", Sigma: " + str(sigma) + ", Train Accuracy: " + str(
        dual.calc_error_gauss(alpha, train_labels, train_features, train_labels, train_features, sigma)) + ", Test Accuracy: " + str(
        dual.calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma)))

    sigma = 0.5
    alpha = dual.optimize_alpha_gauss(train_features, train_labels, 100/873, sigma)
    print("C: " + str(100 / 873) + ", Sigma: " + str(sigma) + ", Train Accuracy: " + str(
        dual.calc_error_gauss(alpha, train_labels, train_features, train_labels, train_features, sigma)) + ", Test Accuracy: " + str(
        dual.calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma)))
    alpha = dual.optimize_alpha_gauss(train_features, train_labels, 500/873, sigma)
    print("C: " + str(500 / 873) + ", Sigma: " + str(sigma) + ", Train Accuracy: " + str(
        dual.calc_error_gauss(alpha, train_labels, train_features, train_labels, train_features, sigma)) + ", Test Accuracy: " + str(
        dual.calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma)))
    alpha = dual.optimize_alpha_gauss(train_features, train_labels, 700/873, sigma)
    print("C: " + str(700 / 873) + ", Sigma: " + str(sigma) + ", Train Accuracy: " + str(
        dual.calc_error_gauss(alpha, train_labels, train_features, train_labels, train_features, sigma)) + ", Test Accuracy: " + str(
        dual.calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma)))

    sigma = 1
    alpha = dual.optimize_alpha_gauss(train_features, train_labels, 100/873, sigma)
    print("C: " + str(100 / 873) + ", Sigma: " + str(sigma) + ", Train Accuracy: " + str(
        dual.calc_error_gauss(alpha, train_labels, train_features, train_labels, train_features, sigma)) + ", Test Accuracy: " + str(
        dual.calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma)))
    alpha = dual.optimize_alpha_gauss(train_features, train_labels, 500/873, sigma)
    print("C: " + str(500 / 873) + ", Sigma: " + str(sigma) + ", Train Accuracy: " + str(
        dual.calc_error_gauss(alpha, train_labels, train_features, train_labels, train_features, sigma)) + ", Test Accuracy: " + str(
        dual.calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma)))
    alpha = dual.optimize_alpha_gauss(train_features, train_labels, 700/873, sigma)
    print("C: " + str(700 / 873) + ", Sigma: " + str(sigma) + ", Train Accuracy: " + str(
        dual.calc_error_gauss(alpha, train_labels, train_features, train_labels, train_features, sigma)) + ", Test Accuracy: " + str(
        dual.calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma)))

    sigma = 5
    alpha = dual.optimize_alpha_gauss(train_features, train_labels, 100/873, sigma)
    print("C: " + str(100 / 873) + ", Sigma: " + str(sigma) + ", Train Accuracy: " + str(
        dual.calc_error_gauss(alpha, train_labels, train_features, train_labels, train_features, sigma)) + ", Test Accuracy: " + str(
        dual.calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma)))
    alpha = dual.optimize_alpha_gauss(train_features, train_labels, 500/873, sigma)
    print("C: " + str(500 / 873) + ", Sigma: " + str(sigma) + ", Train Accuracy: " + str(
        dual.calc_error_gauss(alpha, train_labels, train_features, train_labels, train_features, sigma)) + ", Test Accuracy: " + str(
        dual.calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma)))
    alpha = dual.optimize_alpha_gauss(train_features, train_labels, 700/873, sigma)
    print("C: " + str(700 / 873) + ", Sigma: " + str(sigma) + ", Train Accuracy: " + str(
        dual.calc_error_gauss(alpha, train_labels, train_features, train_labels, train_features, sigma)) + ", Test Accuracy: " + str(
        dual.calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma)))

    sigma = 100
    alpha = dual.optimize_alpha_gauss(train_features, train_labels, 100/873, sigma)
    print("C: " + str(100 / 873) + ", Sigma: " + str(sigma) + ", Train Accuracy: " + str(
        dual.calc_error_gauss(alpha, train_labels, train_features, train_labels, train_features, sigma)) + ", Test Accuracy: " + str(
        dual.calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma)))
    alpha = dual.optimize_alpha_gauss(train_features, train_labels, 500/873, sigma)
    print("C: " + str(500 / 873) + ", Sigma: " + str(sigma) + ", Train Accuracy: " + str(
        dual.calc_error_gauss(alpha, train_labels, train_features, train_labels, train_features, sigma)) + ", Test Accuracy: " + str(
        dual.calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma)))
    alpha = dual.optimize_alpha_gauss(train_features, train_labels, 700/873, sigma)
    print("C: " + str(700 / 873) + ", Sigma: " + str(sigma) + ", Train Accuracy: " + str(
        dual.calc_error_gauss(alpha, train_labels, train_features, train_labels, train_features, sigma)) + ", Test Accuracy: " + str(
        dual.calc_error_gauss(alpha, test_labels, test_features, train_labels, train_features, sigma)))
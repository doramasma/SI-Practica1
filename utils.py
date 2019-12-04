import matplotlib.pyplot as plt
import numpy as np

tol = 0.000000000001

def display_error(bpnn):
    plt.figure('SSE Progress')
    plt.plot(list(range(0, bpnn.number_iterations)), bpnn.sse_list)
    plt.xlabel('Epoch')
    plt.ylabel('SSE')
    plt.grid(True)
    plt.title('Epoch VS SSE')
    plt.show()


def display_accuracy(bpnn):
    plt.figure('Accuracy Progress')
    plt.plot(list(range(0, bpnn.number_iterations)), bpnn.accuracy_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.title('Epoch VS Accuracy')
    plt.show()


def _sigmoid(X):
    def int_sigmoid(x):
        if x < -100:
            return tol
        elif x > 100:
            return 100-tol
        else:
            return 1 / (1 + np.exp(-x))

    vect = np.vectorize(int_sigmoid)
    return vect(X)


def _derivative_sigmoid(p_X):
    return _sigmoid(p_X) * (1 - _sigmoid(p_X))


def _ReLU(X):
    def int_ReLU(x):
        return x * (x > 0)

    vec = np.vectorize(int_ReLU)
    return vec(X)


def _derivative_ReLU(X):
    def int_dReLU(x):
        return 1. * (x > 0)

    vec = np.vectorize(int_dReLU)
    return vec(X)

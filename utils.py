import matplotlib.pyplot as plt
import numpy as np


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

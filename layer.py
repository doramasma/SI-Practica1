import abc
import numpy


def _sigmoid(X):
    def int_sigmoid(x):
        if x < -100:
            return 0
        elif x > 100:
            return 1
        else:
            return 1 / (1 + numpy.exp(-x))
    vect = numpy.vectorize(int_sigmoid)
    return vect(X)


class Layer(object):
    """Class Layer:
    Attributes:
        number_neurons.-
        number_inputs_each_neuron.-
        w.-
    Methods:
         __init__(p_number_neurons, p_number_inputs, p_random_state)
         init_w()
         _net_input(p_X)
         _activation(p_net_input)
         _quantization(p_activation)
         predict(p_X)
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, p_number_neurons=1, p_number_inputs_each_neuron=1):
        self.number_neurons = p_number_neurons
        self.number_inputs_each_neuron = p_number_inputs_each_neuron

    @abc.abstractmethod
    def init_w(self, p_random_seed=numpy.random.RandomState(None)):
        pass

    def _net_input(self, p_X):
        return numpy.matmul(p_X, self.w[1:, :]) + self.w[0, :]

    @abc.abstractmethod
    def predict(self, p_X):
        pass

    @staticmethod
    def _activation(p_net_input):
        return _sigmoid(p_net_input)

    @staticmethod
    def _quantization(p_activation):
        # print(p_activation[p_activation >= 0.5])
        # TODO: Mayor que cero 1 <=> Menor que cero -1
        return numpy.where(p_activation >= 0.5, 1, 0)


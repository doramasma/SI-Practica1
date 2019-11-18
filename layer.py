import abc
import numpy
import math

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
        """
        print("======================================================")
        print("Entradas: ", p_X)
        print("Pesos: \n", self.w[1:,:])
        print("Multiplicación: ", numpy.matmul(p_X, self.w[1:, :]))
        print("Sumamos w0: ", self.w[0, :])
        print("Nueva: ", numpy.matmul(p_X, self.w[1:, :]) + self.w[0, :])
        """
        return numpy.matmul(p_X, self.w[1:, :]) + self.w[0, :] #TODO: que pasa si le sumamos al matmul el peso 0


    def _quantization(self, p_activation):
        #TODO: Mayor que cero 1 <=> Menor que cero -1
        return numpy.where(p_activation >= 0, 1, 0)

    @abc.abstractmethod
    def predict(self, p_X):
        pass

    def _activation(self, p_net_input):
        return self._sigmoid(p_net_input)

    def _sigmoid(self, X):
        sigmoid = lambda x: 1 / (1 + numpy.exp(-x))
        vect = numpy.vectorize(sigmoid)
        return vect(X)

import abc
import numpy

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
        self.last_net_input=0
        self.last_output=0

    @abc.abstractmethod
    def init_w(self, p_random_seed=numpy.random.RandomState(None)):
        pass

    def _net_input(self, p_X):
        return numpy.matmul(p_X, self.w[1:, :]) + self.w[0, :]

    @abc.abstractmethod
    def predict(self, p_X):
        pass

    @abc.abstractmethod
    def _activation(self, p_net_input):
        pass

    @staticmethod
    def _quantization(p_activation):
        # if p_activation.shape[0] == 6974:
        #     numpy.savetxt('activation_predicted_values.csv', p_activation, newline=',')
        return numpy.where(p_activation >= 0.5, 1, 0)

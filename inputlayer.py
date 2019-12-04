import numpy
import layer


class InputLayer(layer.Layer):

    def __init__(self, p_number_neurons=1):
        layer.Layer.__init__(self, p_number_neurons, 1)

    def init_w(self, p_random_seed=numpy.random.RandomState(None)):
        self.w = numpy.concatenate((numpy.zeros((1, self.number_neurons)),
                                    numpy.eye(self.number_neurons)))
        return self

    def predict(self, p_X):
        self.last_net_input = self._net_input(p_X)
        self.last_output = self._activation(self.last_net_input)
        return self.last_output

    def _activation(self, net_input):
        return net_input

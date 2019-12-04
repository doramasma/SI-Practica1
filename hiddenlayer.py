import numpy
import layer
import utils

class HiddenLayer(layer.Layer):

    def init_w(self, p_random_seed=numpy.random.RandomState(None)):
        self.w = p_random_seed.normal(loc=0.0,
                                      scale=.1,
                                      size=(1 + self.number_inputs_each_neuron, self.number_neurons))

        return self.w

    def predict(self, p_X):
        self.last_net_input = self._net_input(p_X)
        self.last_output = self._activation(self.last_net_input)
        return self.last_output

    def _activation(self, net_input):
        return utils._ReLU(net_input)
    
    def _derivative_activation(self):
        return utils._derivative_ReLU(self.last_net_input)

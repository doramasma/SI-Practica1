import numpy
import layer


class OutputLayer(layer.Layer):

    def init_w(self, p_random_seed=numpy.random.RandomState(None)):
        self.w = p_random_seed.normal(loc=0.0,
                                      scale=1,
                                      size=(1 + self.number_inputs_each_neuron, self.number_neurons))
        return self

    def predict(self, p_X):
        return self._quantization(self.activate(p_X))

    def activate(self, p_X):
        return self._activation(self._net_input(p_X))

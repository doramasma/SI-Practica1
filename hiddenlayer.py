import numpy
import layer


class HiddenLayer(layer.Layer):

    def init_w(self, p_random_seed=numpy.random.RandomState(None)):
        self.w = p_random_seed.normal(loc=0.0,
                                      scale=1,
                                      size=(1 + self.number_inputs_each_neuron, self.number_neurons))
        
        return self.w

    def predict(self, p_X):
        return self._activation(self._net_input(p_X))

def _activation(p_net_input):
        return _ReLU(p_net_input)

def _ReLU(X):
    def int_ReLU(x):
        return x * (x > 0)
    vec = numpy.vectorize(int_ReLU)
    return vec(X)
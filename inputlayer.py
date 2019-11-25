import numpy
import layer


# import utils.sigmoid


class InputLayer(layer.Layer):

    def __init__(self, p_number_neurons=1):
        layer.Layer.__init__(self, p_number_neurons, 1)

    def init_w(self, p_random_seed=numpy.random.RandomState(None)):
        self.w = numpy.concatenate((numpy.zeros((1, self.number_neurons)),
                                    numpy.eye(self.number_neurons)))  # TODO: generar filas con valores random
        """
         n0 n1 n2 ...  nx
        [0 0 0 0 0 0 0 0] Peso w0
        [1 0 0 0 0 0 0 0] 1era peso1
        [0 1 0 0 0 0 0 0] 2da peso2
        [0 0 1 0 0 0 0 0] 3era peso3
        ..
        [0 0 0 0 0 0 0 1] xera pesox
        Columnas = neuronas
        filas = pesos
        """

        return self

    def predict(self, p_X):
        return self._net_input(p_X)

    def _activation(self, net_input):
        return net_input

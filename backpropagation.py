import numpy as np
import inputlayer
import hiddenlayer
import outputlayer


def _derivative_sigmoid(p_X):
    def derivative_sigmoid(x): return np.exp(-x) / ((1 + np.exp(-x)) ** 2)
    vect = np.vectorize(derivative_sigmoid)
    return vect(p_X)


class BackPropagation(object):
    """Class BackPropagation:
       Attributes:
         eta.- Learning rate
         number_iterations.-
         ramdon_state.- Random process seed
         input_layer_.-
         hidden_layers_.-
         output_layer_.-
         sse_while_fit_.-
       Methods:
         __init__(p_eta=0.01, p_iterations_number=50, p_ramdon_state=1)
         fit(p_X_training, p_Y_training, p_X_validation, p_Y_validation,
             p_number_hidden_layers=1, p_number_neurons_hidden_layers=np.array([1]))
         predict(p_x) .- Method to predict the output, y
    """

    def __init__(self, p_eta=0.01, p_number_iterations=50, p_random_state=None):
        self.eta = p_eta
        self.number_iterations = p_number_iterations
        self.random_seed = np.random.RandomState(p_random_state)

    def fit(self, p_X_training,
            p_Y_training,
            p_X_validation,
            p_Y_validation,
            batch_size=20,
            p_number_hidden_layers=1,
            p_number_neurons_hidden_layers=np.array([1])):

        self.input_layer_ = inputlayer.InputLayer(p_X_training.shape[1])
        self.hidden_layers_ = []
        for v_layer in range(p_number_hidden_layers):
            if v_layer == 0:
                self.hidden_layers_.append(hiddenlayer.HiddenLayer(p_number_neurons_hidden_layers[v_layer],
                                                                   self.input_layer_.number_neurons))
            else:
                self.hidden_layers_.append(hiddenlayer.HiddenLayer(p_number_neurons_hidden_layers[v_layer],
                                                                   p_number_neurons_hidden_layers[v_layer - 1]))

        self.output_layer_ = outputlayer.OutputLayer(p_Y_training.shape[1],
                                                     self.hidden_layers_[self.hidden_layers_.__len__() - 1].number_neurons)

        self.input_layer_.init_w(self.random_seed)
        for v_hidden_layer in self.hidden_layers_:
            v_hidden_layer.init_w(self.random_seed)
        self.output_layer_.init_w(self.random_seed)

        #TODO: train & validation
        for _ in range(0, self.number_iterations):
            for batch in range(0, p_X_training.shape[0]-batch_size, batch_size):
                current_batch_X = p_X_training[batch:batch + batch_size, :]
                current_batch_Y = p_Y_training[batch:batch + batch_size, :]

                # FORWARD

                v_Y_input_layer_ = self.input_layer_.predict(current_batch_X)
                v_Y_hidden_layer_ = []
                v_Y_hidden_layer_.append(
                    self.hidden_layers_[0].predict(v_Y_input_layer_))
                for v_hiddenlayer in self.hidden_layers_[1:]:
                    v_Y_hidden_layer_.append(
                        self.v_hiddenlayer.predict(v_Y_hidden_layer_[-1]))
                v_Y_output_layer_ = self.output_layer_.activate(
                    v_Y_hidden_layer_[-1])

                # BACKWARD

                # inc_w = self.eta * np.subtract(p_Y_training, v_Y_output_layer_) * _derivative_sigmoid(v_Y_output_layer_) * v_Y_hidden_layer_[-1]
                # inc_w = (np.sum(inc_w, axis= 0)/inc_w.shape[0])
                # print(inc_w.shape)
                # inc_w0 = self.eta * np.subtract(p_Y_training, v_Y_output_layer_)
                # inc_w0 = (np.sum(inc_w0, axis= 0)/inc_w0.shape[0])

                # sigma es el chirimbolito raro (estuve a punto de llamarle LIGMA)
                sigma = self._calculate_sigma_output(current_batch_Y, v_Y_output_layer_)
                delta_w = self._calculate_delta_output(sigma[-1], v_Y_hidden_layer_[-1])
                self.output_layer_.w = self.output_layer_.w + \
                    delta_w.reshape(self.output_layer_.w.shape)
                # previous_sigma es la sigma de la capa a la que se le ha calculado el w en la iteracion anterior (sima de u)
                previous_sigma = (np.sum(sigma, axis=0)/sigma.shape[0])

                for i in reversed(range(len(v_Y_hidden_layer_))):
                    if i > 0:
                        v_Y_next_layer_ = v_Y_hidden_layer_[i - 1]
                    # en el caso en el que sea la primera capa oculta, la capa R de la misma es la capa de input
                    else:
                        v_Y_next_layer_ = v_Y_input_layer_

                    if (i == len(v_Y_hidden_layer_) - 1):
                        previous_w = self.output_layer_.w
                    else:
                        previous_w = self.hidden_layers_[i + 1].w

                    sigma = self._calculate_sigma(previous_sigma, previous_w, v_Y_hidden_layer_[i])
                    print(v_Y_hidden_layer_[i].shape)
                    delta_w = self._calculate_delta(sigma, v_Y_next_layer_)

                    self.hidden_layers_[i].w = self.hidden_layers_[i].w + delta_w.reshape(self.hidden_layers_[i].w.shape)
                    previous_sigma = sigma
                breakpount

        return self

    # def derivative_sigmoid(p_X):
    #     return _sigmoid(p_X) * (1 - _sigmoid(p_X))

    def _calculate_delta_output(self, sigma, v_Y_previous_layer_):
        delta_w = self.eta * sigma * v_Y_previous_layer_
        delta_w = (np.sum(delta_w, axis=0)/delta_w.shape[0])

        delta_w0 = self.eta * sigma
        delta_w0 = (np.sum(delta_w0, axis=0)/delta_w0.shape[0])

        return np.hstack([delta_w0, delta_w])

    def _calculate_sigma_output(self, p_Y_training, v_Y_layer_):
        return np.subtract(p_Y_training, v_Y_layer_) * _derivative_sigmoid(v_Y_layer_)

    def _calculate_delta(self, sigma, v_Y_previous_layer_):
        delta_w = self.eta * sigma * v_Y_previous_layer_
        delta_w = (np.sum(delta_w, axis=0)/delta_w.shape[0])

        delta_w0 = self.eta * sigma
        delta_w0 = (np.sum(delta_w0, axis=0)/delta_w0.shape[0])

        return np.hstack([delta_w0, delta_w])

    def _calculate_sigma(self, previous_sigma, previous_w, v_Y_layer_):
        return np.sum(previous_sigma * previous_w) * _derivative_sigmoid(v_Y_layer_)

    def predict(self, p_X):
        v_Y_input_layer_ = self.input_layer_.predict(p_X)
        v_X_hidden_layer_ = v_Y_input_layer_
        for v_hiddenlayer in self.hidden_layers_:
            v_Y_hidden_layer_ = v_hiddenlayer.predict(v_X_hidden_layer_)
            v_X_hidden_layer_ = v_Y_hidden_layer_
        v_X_output_layer_ = v_Y_hidden_layer_
        v_Y_output_layer_ = self.output_layer_.predict(v_X_output_layer_)
        return v_Y_output_layer_

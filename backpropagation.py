import numpy as np
import inputlayer
import hiddenlayer
import outputlayer


def _derivative_sigmoid(p_X):
    def int_derivative_sigmoid(x):
        if x < -1000:
            return 0
        elif x > 1000:
            return 1
        else:
            return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

    vec = np.vectorize(int_derivative_sigmoid)
    return vec(p_X)


def _sigmoid(X):
    def int_sigmoid(x):
        if x < -1000:
            return 0
        elif x > 1000:
            return 1
        else:
            return 1 / (1 + np.exp(-x))

    vect = np.vectorize(int_sigmoid)
    return vect(X)


def _calculate_sigma_output(p_Y_training, v_Y_layer_):
    return np.subtract(p_Y_training, _sigmoid(v_Y_layer_)) * _derivative_sigmoid(v_Y_layer_)


def _calculate_sigma(previous_sigma, previous_w, v_Y_layer_):
    return np.sum(np.dot(previous_sigma, previous_w.T)) * _derivative_sigmoid(v_Y_layer_)


def get_accuracy(predicted, test):
    n_hits = len([1 for predicted, expected in zip(predicted, test) if predicted == expected])
    return round(n_hits * 100 / len(test), 2)


class BackPropagation(object):
    """Class BackPropagation:
       Attributes:
         eta.- Learning rate
         number_iterations.-
         random_state.- Random process seed
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
            p_batchs_per_epoch=50,
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
                                                     self.hidden_layers_[
                                                         self.hidden_layers_.__len__() - 1].number_neurons)

        self.input_layer_.init_w(self.random_seed)
        for v_hidden_layer in self.hidden_layers_:
            v_hidden_layer.init_w(self.random_seed)
        self.output_layer_.init_w(self.random_seed)

        # TODO: train & validation
        randomize = np.arange(len(p_X_training))  # [0, 1, 2]
        p_X_shuffle = p_X_training.copy()
        p_Y_shuffle = p_Y_training.copy()
        for epoch in range(0, self.number_iterations):
            np.random.shuffle(randomize)  # [2, 1, 0]
            p_X_training = p_X_shuffle[randomize]  # pX[2], pX[1], pX[0]
            p_Y_training = p_Y_shuffle[randomize]  # pY[2], py[1], py[0]
            if p_batchs_per_epoch * batch_size > len(p_X_training):
                p_batchs_per_epoch = int(len(p_X_training) / batch_size)

            for batch in range(0, batch_size * p_batchs_per_epoch, batch_size):
                current_batch_X = p_X_training[batch:batch + batch_size, :]
                current_batch_Y = p_Y_training[batch:batch + batch_size, :]

                # FORWARD
                v_Y_input_layer_ = self.input_layer_.predict(current_batch_X)
                v_Y_hidden_layer_ = [self.hidden_layers_[0].predict(v_Y_input_layer_)]
                for v_hiddenlayer in self.hidden_layers_[1:]:
                    v_Y_hidden_layer_.append(v_hiddenlayer.predict(v_Y_hidden_layer_[-1]))
                v_Y_output_layer_ = self.output_layer_.activate(v_Y_hidden_layer_[-1])
                net_output = self.output_layer_._net_input(v_Y_hidden_layer_[-1])

                # BACKWARD
                sigma = _calculate_sigma_output(current_batch_Y, net_output)
                delta_w = self._calculate_delta(sigma, v_Y_hidden_layer_[-1])
                self.output_layer_.w = self.output_layer_.w + delta_w.reshape(self.output_layer_.w.shape)
                previous_sigma = (np.sum(sigma, axis=0) / sigma.shape[0])

                for i in reversed(range(len(v_Y_hidden_layer_))):
                    if i == len(v_Y_hidden_layer_) - 1 and len(v_Y_hidden_layer_) > 1:
                        previous_w = self.output_layer_.w
                        v_Y_next_layer_ = v_Y_hidden_layer_[i - 1]

                    elif 0 < i < len(v_Y_hidden_layer_) - 1:
                        previous_w = self.hidden_layers_[i + 1].w
                        v_Y_next_layer_ = v_Y_hidden_layer_[i - 1]

                    else:
                        previous_w = self.output_layer_.w if len(v_Y_hidden_layer_) == 1 else self.hidden_layers_[i + 1].w
                        v_Y_next_layer_ = v_Y_input_layer_

                    sigma = _calculate_sigma(previous_sigma, previous_w, v_Y_hidden_layer_[i])
                    delta_w = self._calculate_delta_backward(sigma, v_Y_next_layer_)

                    self.hidden_layers_[i].w = self.hidden_layers_[i].w + delta_w.reshape(self.hidden_layers_[i].w.shape)
                    previous_sigma = sigma

                # TODO: Mirar Capa de Entrada y Pesos

            # TODO: Validation
            if epoch % 5 == 0 or epoch == self.number_iterations-1:
                print("[Iteración", epoch, "]\nSGE  => ", np.mean(np.square(np.subtract(p_Y_validation, self.get_act_value(p_X_validation)))))
                print("Accuracy => ", get_accuracy(self.predict(p_X_validation), p_Y_validation), "\n")

        return self

    def _calculate_delta(self, sigma, v_Y_previous_layer_):
        delta_w = self.eta * np.dot(sigma.T, v_Y_previous_layer_)
        delta_w = (np.sum(delta_w, axis=0) / delta_w.shape[0])

        delta_w0 = self.eta * sigma
        delta_w0 = (np.sum(delta_w0, axis=0) / delta_w0.shape[0])

        return np.hstack([delta_w0, delta_w])

    def _calculate_delta_backward(self, sigma, v_Y_previous_layer_):
        delta_w = self.eta * np.dot(sigma.T, v_Y_previous_layer_)
        delta_w0 = self.eta * sigma
        delta_w0 = np.sum(delta_w0, axis=0) / delta_w0.shape[0]
        return np.vstack([delta_w0.T, delta_w.T])

    def predict(self, p_X):
        v_Y_input_layer_ = self.input_layer_.predict(p_X)
        v_X_hidden_layer_ = v_Y_input_layer_
        for v_hiddenlayer in self.hidden_layers_:
            v_Y_hidden_layer_ = v_hiddenlayer.predict(v_X_hidden_layer_)
            v_X_hidden_layer_ = v_Y_hidden_layer_
        v_X_output_layer_ = v_Y_hidden_layer_
        v_Y_output_layer_ = self.output_layer_.predict(v_X_output_layer_)
        return v_Y_output_layer_

    def get_act_value(self, p_X):
        v_Y_input_layer_ = self.input_layer_.predict(p_X)
        v_X_hidden_layer_ = v_Y_input_layer_
        for v_hiddenlayer in self.hidden_layers_:
            v_Y_hidden_layer_ = v_hiddenlayer.predict(v_X_hidden_layer_)
            v_X_hidden_layer_ = v_Y_hidden_layer_
        v_X_output_layer_ = v_Y_hidden_layer_
        v_Y_output_layer_ = self.output_layer_.activate(v_X_output_layer_)
        return v_Y_output_layer_

import numpy as np
import inputlayer
import hiddenlayer
import outputlayer

def calculate_output_δ(output_layer, p_Y):
    return np.subtract(p_Y, output_layer.last_output) * output_layer._derivative_activation()

def calculate_δ(current_layer, last_δ, last_W):
    return np.dot(last_δ, last_W.T[:,1:]) * current_layer._derivative_activation()

def calculate_ΔW(previous_layer, eta, δ):
    ΔW_no_w0 = eta * np.dot(δ.T, previous_layer.last_output)
    ΔW0 = eta * δ.T
    ΔW = np.hstack([ΔW0, ΔW_no_w0])
    return ΔW

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
        self.sse_list = []
        self.accuracy_list = []

    def fit(self, p_X_training,
            p_Y_training,
            p_X_validation,
            p_Y_validation,
            p_batchs_per_epoch=50,
            p_number_hidden_layers=1,
            p_number_neurons_hidden_layers=np.array([1])):

        print("Porcentaje de accidentes: ", round(np.mean(p_Y_training[:, 0]) * 100, 2), "%\n")

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

        randomize = np.arange(len(p_X_training))
        p_X_shuffle = p_X_training.copy()
        p_Y_shuffle = p_Y_training.copy()
        for epoch in range(0, self.number_iterations):
            np.random.shuffle(randomize)
            p_X_training = p_X_shuffle[randomize]  # pX[2], pX[1], pX[0]
            p_Y_training = p_Y_shuffle[randomize]  # pY[2], py[1], py[0]
            if p_batchs_per_epoch > len(p_X_training):
                p_batchs_per_epoch = len(p_X_training)

            for batch in range(0, p_batchs_per_epoch):
                current_batch_X = p_X_training[batch:batch + 1, :]
                current_batch_Y = p_Y_training[batch:batch + 1, :]

                self.forward_pass(current_batch_X)
                self.backward_pass(current_batch_Y)
            self.show_progress(epoch, p_X_validation, p_Y_validation)
        return
    
    def forward_pass(self, p_X):
        return self.predict(p_X)

    def backward_pass(self, p_Y):
        δ = calculate_output_δ(self.output_layer_, p_Y)
        ΔW = calculate_ΔW(self.hidden_layers_[-1], self.eta, δ)
        

        self.output_layer_.w += ΔW.T
        last_W = self.output_layer_.w
        last_δ = δ
        for i in reversed(range(0, len(self.hidden_layers_))):
            if i == 0:
                previous_layer = self.input_layer_
            else:
                previous_layer = self.hidden_layers_[i-1]

            δ = calculate_δ(self.hidden_layers_[i], last_δ, last_W)
            ΔW = calculate_ΔW(previous_layer, self.eta, δ)

            self.hidden_layers_[i].w += ΔW.T
            last_W = self.hidden_layers_[i].w
            last_δ = δ
            
        return

    def show_progress(self, epoch, p_X_validation, p_Y_validation):
        self.predict(p_X_validation)
        sse = np.mean(np.square(np.subtract(p_Y_validation, self.output_layer_.last_output)))
        accuracy = get_accuracy(self.predict(p_X_validation), p_Y_validation)
        self.sse_list.append(sse)
        self.accuracy_list.append(accuracy)
        print("\rIteracion %s: ACC: %f\tSSE: %f\t\t\t" % (epoch, accuracy, sse), end='')

    def predict(self, p_X):
        v_Y_input_layer_ = self.input_layer_.predict(p_X)
        v_X_hidden_layer_ = v_Y_input_layer_
        for v_hiddenlayer in self.hidden_layers_:
            v_Y_hidden_layer_ = v_hiddenlayer.predict(v_X_hidden_layer_)
            v_X_hidden_layer_ = v_Y_hidden_layer_
        v_X_output_layer_ = v_Y_hidden_layer_
        v_Y_output_layer_ = self.output_layer_.predict(v_X_output_layer_)
        return v_Y_output_layer_
import numpy
import inputlayer
import hiddenlayer
import outputlayer


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
         fit(p_X_training, p_Y_training, p_X_validation, p_Y_validation, p_number_hidden_layers=1, p_number_neurons_hidden_layers=numpy.array([1]))
         predict(p_x) .- Method to predict the output, y
    """

    def __init__(self, p_eta=0.01, p_number_iterations=50, p_random_state=None):
        self.eta = p_eta
        self.number_iterations = p_number_iterations
        self.random_seed = numpy.random.RandomState(p_random_state)

    def fit(self, p_X_training,
                  p_Y_training,
                  p_X_validation,
                  p_Y_validation,
                  p_number_hidden_layers=1,
                  p_number_neurons_hidden_layers=numpy.array([1])):

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
            v_Y_input_layer_ = self.input_layer_.predict(p_X)
            v_Y_hidden_layer_ = []
            v_Y_hidden_layer_.append(self.v_hidden_layers[0].predict(v_Y_input_layer_))
            for v_hiddenlayer in self.hidden_layers_[1:]:
                v_Y_hidden_layer_.append(self.v_hidden_layer.predict(v_Y_hidden_layer_[-1]))
            v_Y_output_layer_ = self.output_layer_.activate(v_Y_hidden_layer_[-1])

            # error = f_energy(v_Y_output_layer_, p_Y_training)
            
            self.output_layer_.w += self.eta * numpy.substract(p_Y, v_Y_output_layer_) * * v_Y_hidden_layer_[-1]
            
            #for i-1 in reversed(range((v_Y_hidden_layer_))):
                #self.v_Y_hidden_layer_[i]  = 
            


        return self

    def derivative_sigmoid(p_X):
        return _sigmoid(p_X)*(1 - _sigmoid(p_X)

    def predict(self, p_X):
        v_Y_input_layer_ = self.input_layer_.predict(p_X)
        v_X_hidden_layer_ = v_Y_input_layer_
        for v_hiddenlayer in self.hidden_layers_:
            v_Y_hidden_layer_ = v_hiddenlayer.predict(v_X_hidden_layer_)
            v_X_hidden_layer_ = v_Y_hidden_layer_
        v_X_output_layer_ = v_Y_hidden_layer_
        v_Y_output_layer_ = self.output_layer_.predict(v_X_output_layer_)
        return v_Y_output_layer_
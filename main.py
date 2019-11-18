import backpropagation
import numpy
import pandas
from datetime import datetime as dt

TIPO_PRECIPITACION = {'TIPO_PRECIPITACION':{'clear':0, 'rain':1, 'snow':3 }}

INTENSIDAD_PRECIPITACION = {'INTENSIDAD_PRECIPITACION':{'High':0, 'Low':1, 'Moderate':2, 'None':3}} 

ESTADO_CARRETERA = {'ESTADO_CARRETERA':{'Dry':0, 'Snow covered':1, 'Visible tracks':2, 'Wet':3}}

ACCIDENTE = {'ACCIDENTE': {'No': 0, 'Yes': 1}}

listDict = [TIPO_PRECIPITACION, INTENSIDAD_PRECIPITACION, ESTADO_CARRETERA, ACCIDENTE]

class bcolors:
    ACCURACY = '\033[92m'
    ERROR = '\033[91m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    LOADING= '\x1b[7;30;46m'
    TRAINING= '\x1b[7;30;46m'
    RESULT= '\x1b[7;30;47m'
    LOADING2= '\x1b[6;30;42m'

def process_input():
    # TODELETE Fecha y hora 0, Tipo de precipitacion 8, intensidad_precipitacion 9, estado carretera 12, accidentes 13
    df = pandas.read_excel ('./dataset/dataset_assignment_1.xls', na_values=[' '])
    df = df.dropna()

    df['FECHA_HORA'] = [x.replace(',', '.') for x in df['FECHA_HORA']]
    df['FECHA_HORA'] = [float(x.replace(x, str(dt.fromisoformat(x).timestamp()))) for x in df['FECHA_HORA']]
    
    for _dict in listDict: df = df.replace(_dict)    

    return df

def df_split(df, test_range=.10):
    total_size = df.shape[0]
    validation_size = round(total_size * test_range)
    train_size = total_size - validation_size

    train_df =  numpy.array(df.iloc[0:train_size, :-1])
    test_df =  numpy.array(df.iloc[train_size:, :-1])
    train_df_outputs = numpy.zeros((train_size, 1))
    test_df_outputs = numpy.zeros((validation_size, 1))
    
    for index, val in enumerate(numpy.array(df.iloc[:, -1])):
        if (index < train_size):
            train_df_outputs[index] = val
        else: 
            test_df_outputs[index - train_size] = val

    return train_df, test_df, train_df_outputs, test_df_outputs 

def norm_dataframe(df):
    normalized_df = (df - df.mean())/df.std()
    return numpy.array(normalized_df)

def norm_df(df):
    x = pandas.DataFrame( df.iloc[:,:-1] )
    norm_x = numpy.array((x - x.mean())/x.std())
    norm_x[:,-1]=df.iloc[:,-1]
    normalized_df = pandas.DataFrame( norm_x )
    return normalized_df


def get_accuracy(predicted, test):
    n_hits = len([1 for predicted, expected in zip(predicted, test) if predicted == expected])
    return round(n_hits*100/len(test),2)

if __name__ == "__main__":
    print((bcolors.LOADING + bcolors.BOLD) + "|============[Getting info from dataset...]============|" + bcolors.ENDC)
    df = process_input()

    print(norm_df(df))
    
    p_X_training, p_X_test, p_Y_training, p_Y_test = df_split(df, 0.10)
    p_X_training, p_X_validation, p_Y_training, p_Y_validation = df_split(pandas.DataFrame(numpy.concatenate((p_X_training, p_Y_training), axis=1)), 0.15)

    p_X_training = norm_dataframe(pandas.DataFrame(p_X_training))
    p_X_validation = norm_dataframe(pandas.DataFrame(p_X_validation))
    p_X_test = norm_dataframe(pandas.DataFrame(p_X_test))

    # print("Training dataset: ", df.shape)
    # print("Training dataset: ", p_X_training.shape)
    # print("test dataset: ", p_X_test.shape)
    # print("Validation dataset : ", p_X_validation.shape)

    print((bcolors.LOADING + bcolors.BOLD) + "|===============[End of input process]=================|" + bcolors.ENDC)

    print("\n" + (bcolors.LOADING + bcolors.BOLD)  + "|=================[Training BPNN...]===================|" + bcolors.ENDC)
    backpropagation = backpropagation.BackPropagation(p_eta=0.01, p_number_iterations=10, p_random_state=1)

    backpropagation.fit(p_X_training = p_X_training,
                        p_Y_training = p_Y_training,
                        p_X_validation = p_X_validation,
                        p_Y_validation = p_Y_validation,
                        p_number_hidden_layers = 1,
                        p_number_neurons_hidden_layers=numpy.array([3]))

    print((bcolors.LOADING + bcolors.BOLD) + "|====================[BPNN trained]====================|" + bcolors.ENDC)

    print("\n" + (bcolors.LOADING + bcolors.BOLD) + "|==============[Predicting new values...]==============|" + bcolors.ENDC)
    predict = backpropagation.predict(p_X_test)
    accuracy = get_accuracy(predict, p_Y_test)
    print((bcolors.LOADING + bcolors.BOLD) + "|==================[Values predicted]==================|" + bcolors.ENDC)
    
    print("\n" + (bcolors.RESULT + bcolors.BOLD) + "|================[Printing results...]=================|" + bcolors.ENDC)
    print((bcolors.LOADING + bcolors.BOLD)  + "Accuracy: " + bcolors.ENDC + bcolors.ACCURACY + str(accuracy) + " %" + bcolors.ENDC)
    #TODO Estoy doramio hente -> Accuracy
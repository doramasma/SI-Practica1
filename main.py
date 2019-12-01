import backpropagation
import numpy as np
import pandas as pd
from datetime import datetime as dt

TIPO_PRECIPITACION = {'TIPO_PRECIPITACION': {'clear': 0, 'rain': 1, 'snow': 2}}

INTENSIDAD_PRECIPITACION = {'INTENSIDAD_PRECIPITACION': {'High': 0, 'Low': 1, 'Moderate': 2, 'None': 3}}

ESTADO_CARRETERA = {'ESTADO_CARRETERA': {'Dry': 0, 'Snow covered': 1, 'Visible tracks': 2, 'Wet': 3}}

ACCIDENTE = {'ACCIDENTE': {'No': 0, 'Yes': 1}}

listDict = [TIPO_PRECIPITACION, INTENSIDAD_PRECIPITACION, ESTADO_CARRETERA, ACCIDENTE]


class BColors:
    ACCURACY = '\033[92m'
    ERROR = '\033[91m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    LOADING = '\x1b[7;30;46m'
    TRAINING = '\x1b[7;30;46m'
    RESULT = '\x1b[7;30;47m'
    LOADING2 = '\x1b[6;30;42m'


def process_input():
    dataset = pd.read_excel('./dataset/dataset_assignment_1.xls', na_values=[' '])
    dataset = dataset.dropna()

    dataset['FECHA_HORA'] = [x.replace(',', '.') for x in dataset['FECHA_HORA']]
    dataset['FECHA_HORA'] = [float(x.replace(x, str(dt.fromisoformat(x).timestamp()))) for x in dataset['FECHA_HORA']]

    for _dict in listDict:
        dataset = dataset.replace(_dict)

    return dataset


def df_split(dataset, test_range=.10):
    total_size = dataset.shape[0]
    validation_size = round(total_size * test_range)
    train_size = total_size - validation_size

    train_df = np.array(dataset.iloc[0:train_size, :-1])
    test_df = np.array(dataset.iloc[train_size:, :-1])
    train_df_outputs = np.zeros((train_size, 1))
    test_df_outputs = np.zeros((validation_size, 1))

    for index, val in enumerate(np.array(dataset.iloc[:, -1])):
        if index < train_size:
            train_df_outputs[index] = val
        else:
            test_df_outputs[index - train_size] = val

    return train_df, test_df, train_df_outputs, test_df_outputs


def norm_dataframe(dataset):
    normalized_df = (dataset - dataset.mean()) / dataset.std()
    return np.array(normalized_df)


def norm_df(dataset):
    x = pd.DataFrame(dataset.iloc[:, :-1])
    norm_x = np.array((x - x.mean()) / x.std())
    norm_x[:, -1] = dataset.iloc[:, -1]
    normalized_df = pd.DataFrame(norm_x)
    return normalized_df


def get_accuracy(predicted, test):
    n_hits = len([1 for predicted, expected in zip(predicted, test) if predicted == expected])
    return round(n_hits * 100 / len(test), 2)


if __name__ == "__main__":
    print((BColors.LOADING + BColors.BOLD) + "|============[Getting info from dataset...]============|" + BColors.ENDC)
    df = process_input()

    p_X_training, p_X_test, p_Y_training, p_Y_test = df_split(df, 0.10)
    p_X_training, p_X_validation, p_Y_training, p_Y_validation = df_split(
        pd.DataFrame(np.concatenate((p_X_training, p_Y_training), axis=1)), 0.15)

    normalizado1 = norm_dataframe(pd.DataFrame(p_X_training[:,0:8]))
    normalizado2 = norm_dataframe(pd.DataFrame(p_X_training[:,10:12]))
    
    p_X_training = np.array(pd.concat([pd.DataFrame(normalizado1),
                             pd.DataFrame(p_X_training[:,8:10]),
                             pd.DataFrame(normalizado2),
                             pd.DataFrame(p_X_training[:,12])],axis=1, ignore_index=True).iloc[:,:])
    
    
    print(p_X_training)

    p_X_validation = norm_dataframe(pd.DataFrame(p_X_validation))
    p_X_test = norm_dataframe(pd.DataFrame(p_X_test))
    p_X_crash_test = np.array(df.loc[df['ACCIDENTE'] == 1].iloc[:,:-1])
    p_Y_crash_test = np.zeros((len(p_X_crash_test), 1))
    
    #Printeos para ver los valores que hay despues de la normalizacion en las columnas con valores categoricos (creo)
    print("La columna 8 = Tipo_Precipitacion")
    print(np.unique(np.array(p_X_training[:,8])))
    print("La columna 9 = Intensidad_Precipitacion")
    print(np.unique(np.array(p_X_training[:,9])))
    print("La columna 12 = Estado_Carretera")
    print(np.unique(np.array(p_X_training[:,12])))
    
    for index, val in enumerate(np.array(df.loc[df['ACCIDENTE'] == 1].iloc[:,-1])):
        p_Y_crash_test[index] = val

    # iris_dataset = pd.read_csv("dataset/iris.csv")
    # x_pre = list()
    # for i in range(0, 100):
    #     x_pre.append([iris_dataset['petal.length'][i], iris_dataset['petal.width'][i]])
    # p_X = np.array(x_pre)
    # p_Y = np.zeros((len(p_X), 1))

    # for i in range(0, 100):
    #     if i < 50:
    #         p_Y[i] = np.zeros(1)
    #     else:
    #         p_Y[i] = np.ones(1)

    # randomize = np.arange(len(p_X))
    # np.random.shuffle(randomize)
    # p_X = p_X[randomize]
    # p_Y = p_Y[randomize]

    # p_X_training, p_X_validation, p_X_test = p_X[0:70], p_X[71:90], p_X[91:100]
    # p_Y_training, p_Y_validation, p_Y_test = p_Y[0:70], p_Y[71:90], p_Y[91:100]

    # print((BColors.LOADING + BColors.BOLD) + "|===============[End of input process]=================|" + BColors.ENDC)

    print("\n" + (BColors.LOADING + BColors.BOLD) + "|=================[Training BPNN...]===================|" + BColors.ENDC)
    backpropagation = backpropagation.BackPropagation(p_eta=0.001, p_number_iterations=200, p_random_state=1)

    backpropagation.fit(p_X_training=p_X_training,
                        p_Y_training=p_Y_training,
                        p_X_validation=p_X_validation,
                        p_Y_validation=p_Y_validation,
                        batch_size=1,
                        p_batchs_per_epoch=100,
                        p_number_hidden_layers=1,
                        p_number_neurons_hidden_layers=np.array([16]))

    # print((BColors.LOADING + BColors.BOLD) + "|====================[BPNN trained]====================|" + BColors.ENDC)

    print("\n" + (BColors.LOADING + BColors.BOLD) + "|=======[Predicting new values (Random Test)...]=======|" + BColors.ENDC)
    # predict = backpropagation.predict(p_X_test)
    # accuracy = get_accuracy(predict, p_Y_test)
    # print((BColors.LOADING + BColors.BOLD) + "|==================[Values predicted]==================|" + BColors.ENDC)

    # print("\n" + (BColors.RESULT + BColors.BOLD) + "|================[Printing results...]=================|" + BColors.ENDC)
    # print((BColors.LOADING + BColors.BOLD) + "Accuracy: " + BColors.ENDC + BColors.ACCURACY + str(accuracy) + " %" + BColors.ENDC)

    print("\n" + (BColors.LOADING + BColors.BOLD) + "|========[Predicting new values (Crash Test)...]=======|" + BColors.ENDC)
    predict = backpropagation.predict(p_X_crash_test)
    accuracy = get_accuracy(predict, p_Y_crash_test)

    # print((BColors.LOADING + BColors.BOLD) + "|==================[Values predicted]==================|" + BColors.ENDC)

    print("\n" + (BColors.RESULT + BColors.BOLD) + "|================[Printing results...]=================|" + BColors.ENDC)
    print((BColors.LOADING + BColors.BOLD) + "Accuracy: " + BColors.ENDC + BColors.ACCURACY + str(accuracy) + " %" + BColors.ENDC)

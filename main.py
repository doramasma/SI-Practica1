import backpropagation
import numpy
import pandas
from datetime import datetime as dt

TIPO_PRECIPITACION = {'TIPO_PRECIPITACION':{'clear':0, 'rain':1, 'snow':3 }}

INTENSIDAD_PRECIPITACION = {'INTENSIDAD_PRECIPITACION':{'High':0,'Low':1,'Moderate':2,'None':3}} 

ESTADO_CARRETERA = {'ESTADO_CARRETERA':{'Dry':0, 'Snow covered':1, 'Visible tracks':2, 'Wet':3}}

ACCIDENTE = {'ACCIDENTE': {'No': 0, 'Yes': 1}}

listDict = [TIPO_PRECIPITACION, INTENSIDAD_PRECIPITACION, ESTADO_CARRETERA, ACCIDENTE]


def process_input():
    # TODELETE Fecha y hora 0, Tipo de precipitacion 8, intensidad_precipitacion 9, estado carretera 12, accidentes 13
    df = pandas.read_excel ('./dataset/dataset_assignment_1.xls', na_values=[' '])
    df = df.dropna()

    df['FECHA_HORA'] = [x.replace(',', '.') for x in df['FECHA_HORA']]
    df['FECHA_HORA'] = [float(x.replace(x, str(dt.fromisoformat(x).timestamp()))) for x in df['FECHA_HORA']]
    
    for _dict in listDict: df = df.replace(_dict)    
    return df


if __name__ == "__main__":


    df = process_input()
    p_X_training = numpy.array(df.loc[:, df.columns != 'ACCIDENTE'].values)

    p_Y_training = numpy.zeros((df.shape[0], 1))
    for i in range(0, len(df['ACCIDENTE'].values)):
        p_Y_training[i] = numpy.array(df['ACCIDENTE'].values[i])

    backpropagation = backpropagation.BackPropagation(p_eta=0.01, p_number_iterations=10, p_random_state=1)

    backpropagation.fit(p_X_training = p_X_training,
                        p_Y_training = p_Y_training,
                        p_X_validation = p_X_training,
                        p_Y_validation = p_Y_training,
                        p_number_hidden_layers=1,
                        p_number_neurons_hidden_layers=numpy.array([3]))

    predict_X = numpy.array(p_X_training)
    predict = backpropagation.predict(predict_X)
    #print(numpy.where(predict==-1))



from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from google.cloud import storage
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import joblib
from RainPredictionMachine.data import CleanDataRpm
from tensorflow.keras.layers import LSTM, Dense
### GCP configuration - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'rain-prediction-machine' # GCP Storage
MODEL_NAME = 'rpmodel' # model folder name (will contain the folders for all trained model versions)
MODEL_VERSION = 'v1_vitor_isa' # model version folder name (where the trained model.joblib file will be stored)


def upload_model_to_gcp(file):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'model_thami/{file}')
    blob.upload_from_filename(file)

def subsample_sequence(df, length):
    index = np.random.randint(0, df.shape[0] - length)
    df_sample = df.iloc[index:index+length]

    return df_sample

def split_subsample_sequence(df, length):
    '''Create one single random (X,y) pair'''

    df_subsample = subsample_sequence(df, length)
    y_sample = df_subsample['Chuva'].iloc[length - 24:]

    X_sample = df_subsample.drop(columns= ['classe_chuva', 'datahora'])[0:length -24]
    X_sample = X_sample.values

    return np.array(X_sample), np.array(y_sample)

def get_X_y(df, n_sequences, length):
    '''Return a list of samples (X, y)'''
    X, y = [], []

    for i in range(n_sequences):
        (xi, yi) = split_subsample_sequence(df, length)
        X.append(xi)
        y.append(yi)

    X = np.array(X)
    y = np.array(y)
    return X, y

#-------split the data--------------
def split_train_test(df, n_sequences, length):
    train_size = int(df.shape[0]*0.8)
    df_train = df.iloc[:train_size,:]
    df_test = df.iloc[train_size:,:]
    X_train, y_train = get_X_y(df_train, n_sequences, length)
    X_test, y_test =  get_X_y(df_test, n_sequences, length)
    return X_train, y_train, X_test, y_test

#-------initialize model-----------------------
def init_model(X_train):
    norm = Normalization()
    norm.adapt(X_train)
    model = Sequential()
    model.add(norm)

    model.add(LSTM(units=200, activation='tanh',return_sequences=True))
    model.add(LSTM(units=100, activation='tanh',return_sequences=True))
    model.add(LSTM(units=50, activation='tanh',return_sequences=False))
    model.add(Dense(30, activation="tanh"))
    model.add(Dense(24, activation="relu"))

    model.compile(loss='mape',
                  optimizer='rmsprop',
                  metrics=['mae','mse','mape'])

    model.compile(loss='mse',
                    optimizer='rmsprop',
                    metrics=['mae','mse','mape'])
    return model

#-------fit model and early stopping-------------
def fit_model(model, X_train, y_train):
    es = EarlyStopping(patience=50, restore_best_weights=True, monitor='val_mape')

    model.fit(X_train, y_train, batch_size=32, epochs=200, verbose=1,
         validation_split=0.2,
          callbacks=[es])
    return model

if __name__ == "__main__":
    #----------------fetch the dataset----------------
    # criar dataframe com os dados tratados a partir da classe
    from RainPredictionMachine.data import CleanDataRpm
    cleaner = CleanDataRpm()
    for index,estacao in enumerate(cleaner.cidades):
        df = cleaner.clean_data(index, gcp=True)
        print('arquivos carregados')
        #split do treino e test
        X_train, y_train, X_test, y_test = split_train_test(df, 6000,72)
        print('split treino e teste')
        #criando o modelo ate o compile
        model = init_model(X_train)
        print('criando o modelo ate o compile')
        #fit e early stopping
        model = fit_model(model, X_train, y_train)
        print('fit e early stopping')
        joblib.dump(model, f'{estacao}.joblib')
        upload_model_to_gcp(f'{estacao}.joblib')

from sklearn.pipeline import Pipeline
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Flatten, LSTM
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from google.cloud import storage
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import joblib
### GCP configuration - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'rain-prediction-machine' # GCP Storage
MODEL_NAME = 'rpmodel' # model folder name (will contain the folders for all trained model versions)
MODEL_VERSION = 'v1_vitor_isa' # model version folder name (where the trained model.joblib file will be stored)
def upload_model_to_gcp(file):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'models/{file}')
    blob.upload_from_filename(file)







if __name__ == "__main__":
    #----------------fetch the dataset----------------
    import pandas as pd
    # criar dataframe com os dados tratados a
    # partir da classe (possivelmente importar outro pacote para incluir no dataframe)
    from RainPredictionMachine.data import CleanDataRpm
    cleaner = CleanDataRpm()
    df = cleaner.clean_data(5, gcp=True)
    print('arquivos carregados')
    #pipe_treinado = pipe_creator(df)
    #print('pipe treinado')
    joblib.dump(df, 'df5.joblib')
    upload_model_to_gcp('df5.joblib')
    #joblib.dump(pipe_treinado, 'model.joblib')
    #upload_model_to_gcp('model.joblib')

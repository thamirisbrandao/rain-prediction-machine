from sklearn.pipeline import Pipeline
from tensorflow.keras import Sequential
<<<<<<< HEAD
from tensorflow.keras.layers import LSTM, Dense

from tensorflow.keras.layers.experimental.preprocessing import Normalization
#class Normalizing(BaseEstimator, TransformerMixin):
#    def __init__(self,feature_name,additional_param=None):
#        self.feature_name = feature_name
#        self.additional_param = additional_param
#    def fit(self,X,y = None):
#        return self
#    def transform(self,X,y=None):
#        X_ = X.copy()
#        normalizer = Normalization()
#        normalizer.adapt(X_[self.feature_name])
#        normalizer(X_[self.feature_name])[0]
#        return X_
#normalizing_pipe = Pipeline([
#    ('Normalizing', Normalizing()),
#])

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Masking
=======
from tensorflow.keras.layers import Dense, SimpleRNN, Flatten, LSTM
from tensorflow.keras.layers.experimental.preprocessing import Normalization
>>>>>>> 643db615548bb09da3e741b63dc2b5fc3c3d23d7
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
<<<<<<< HEAD
 #   joblib.dump(pipe_treinado, 'model.joblib')
 #   upload_model_to_gcp('model.joblib')
=======
    #joblib.dump(pipe_treinado, 'model.joblib')
    #upload_model_to_gcp('model.joblib')
>>>>>>> 643db615548bb09da3e741b63dc2b5fc3c3d23d7

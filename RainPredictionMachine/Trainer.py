#----------------create custom transformer----------------
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd

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
#class Padding_masking(BaseEstimator, TransformerMixin):
#    def __init__(self,feature_name,additional_param=None):
#        self.feature_name = feature_name
#        self.additional_param = additional_param
#    def fit(self,X,y = None):
#        return self
#    def transform(self,X,y=None):
#        X_ = X.copy()
#        X_[self.feature_name] = Masking(X_[self.feature_name],value=-10000)
#        X_[self.feature_name] = pad_sequences(X_[self.feature_name], padding='post', value=-10000)
#        return X_


def pipe_creator(df):
    '''
    this function gets all the data, cleans it and inserts into the pipeline for RNN model fitting.
    '''
    #----------------Hold out----------------
    #separando X de y e depois separando os dados de ajuste dos dados de teste
    from sklearn.model_selection import train_test_split
    X = df.copy()
    y = pd.DataFrame(df['classe_chuva'])

    #----------------label encoder----------------
    #fazendo encoding de variáveis categóricas
    from sklearn.preprocessing import LabelEncoder
    label = LabelEncoder()
    y_enc = label.fit_transform(y)
    from tensorflow.keras.utils import to_categorical
    y_cat = to_categorical(y_enc)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3) #separando train e test

    #----------------scaling pipeline----------------
    #fazendo escalonamento
    from sklearn.preprocessing import StandardScaler
    scaling_pipe = Pipeline([
        ('stdscaler', StandardScaler()),
    ])

    #----------------ordinal encoding pipeline----------------
    #fazendo encoding de variáveis categóricas
    from sklearn.preprocessing import OrdinalEncoder
    ordinal_encoding_pipe = Pipeline([
        ('ordinal_encoding', OrdinalEncoder()),
    ])

    #----------------onehot encoding pipeline----------------
    #fazendo encoding de variáveis categóricas
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoding_pipe = Pipeline([
        ('onehot_encoding', OneHotEncoder()),
    ])

    #----------------padding and masking pipe----------------
    #fazendo encoding de variáveis categóricas
    #Padding and masking class

    #padding_masking_pipe = Pipeline([
    #    ('Padding and masking', Padding_masking()),
    #])

    #----------------column transformer----------------
    #realizando as operações em paralelo
    from sklearn.compose import ColumnTransformer
    col_trans = ColumnTransformer([
        ('encoding', onehot_encoding_pipe , ['classe_chuva']),
        ('scaling ', scaling_pipe,[ 'Pres',
                                    'Pres_max',
                                    'Pres_min',
                                    'Radiacao',
                                    'Temp',
                                    'Temp_orvalho',
                                    'Temp_max',
                                    'Temp_min',
                                    'Temp_orvalho_max',
                                    'Temp_orvalho_min',
                                    'Umid_max',
                                    'Umid_min',
                                    'Umid',
                                    'Rajada_vento',
                                    'Vel_vento',
                                    'Chuva'])
    ])

    #----------------total pipeline----------------
    #concatenar treinamento do modelo
    #full_pipe = Pipeline([
    #    ('column stransformer', col_trans),
    #    ( , ),
    #])

    #----------------GridSearch pipeline----------------
    #aplicar gridsearch
    #from sklearn.model_selection import GridSearchCV
    #grid_search = GridSearchCV(
    #    full_pipe,
    #    param_grid={
    #        'imputer__n_neighbors': [2,5,10]},
    #        cv=5,
    #    scoring="recall")
    #instanciar pipe
    #aplicar

    X_train_t = col_trans.fit_transform(X_train)
    X_test_t = col_trans.transform(X_test)
    return X_train_t,X_test_t


if __name__ == "__main__":
        #----------------fetch the dataset----------------
    import pandas as pd
    from RainPredictionMachine.data import clean_data_rpm
    df = clean_data_rpm()
    df = df.clean_data(1)
    # criar dataframe com os dados tratados a
    # partir da classe (possivelmente importar outro pacote para incluir no dataframe)
    X_train_t, X_test_t = train_data(df)
    joblib.dump(self.pipeline, 'model.joblib')

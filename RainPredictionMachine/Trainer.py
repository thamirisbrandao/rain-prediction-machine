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
                                    #'Chuva',

    ])

    def RNN_model():

        ###########################
        # 1. Define architecture  #
        ###########################
            # Notice that we don't specify the input shape yet...
            # ... as we don't know the shape post-preprocessing!
            # One consequence is that here, you cannot yet print
            # the model's summary. It will be known after fitting it
            # to X_train_preprocessed, y_train

        norm = Normalization()
        model = Sequential()
        model.add(norm)
        model.add(LSTM(units=20, activation='tanh'))
        model.add(Dense(10, activation="tanh"))
        model.add(Dense(4, activation="softmax"))

        model.compile(loss='categorical_crossentropy',
                        optimizer='rmsprop',
                        metrics=['accuracy'])

        ###########################
        # 2. Compile model        #
        ###########################
        model.compile(loss = 'binary_crossentropy',
                    optimizer = 'adam',
                    metrics = ['accuracy'])

        return model

    #from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
    #RNN_model = KerasClassifier(build_fn = RNN_model(),
    #                                   epochs = 10000,
    #                                   batch_size = 32,
    #                                   verbose = 1)

    #----------------total pipeline----------------
    #concatenar treinamento do modelo
    full_pipe = Pipeline([
        ('column_stransformer', col_trans),
    #    ("deep_learning" , RNN_model ),
    ])

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

<<<<<<< HEAD
    #full_pipe.fit(X_train)
    full_pipe.fit(X_train)

=======

    #full_pipe.fit(X_train)
    full_pipe.fit(X_train)
>>>>>>> 3f29fe4debe151320174930323b786b74a1d6fcb
    return full_pipe, y_train


if __name__ == "__main__":
    #----------------fetch the dataset----------------
    import pandas as pd
    # criar dataframe com os dados tratados a
    # partir da classe (possivelmente importar outro pacote para incluir no dataframe)
    from RainPredictionMachine.data import CleanDataRpm
    clean_data = CleanDataRpm()
    df = clean_data.clean_data(1)
    pipe_treinado = pipe_creator(df)
    joblib.dump(pipe_treinado, 'model.joblib')

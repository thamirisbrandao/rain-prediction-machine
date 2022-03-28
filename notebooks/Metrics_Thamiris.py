#!/usr/bin/env python
# coding: utf-8

# In[22]:


import joblib
import pandas as pd
from RainPredictionMachine.Trainer import *
from google.cloud import storage
import os


# In[180]:


def download_model(estacao,bucket='rain-prediction-machine', rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = f'models/{estacao}.joblib'
    blob = client.blob(storage_location)
    blob.download_to_filename(f'{estacao}.joblib')
    print("=> estacao downloaded from storage")
    model = joblib.load(f'{estacao}.joblib')
    if rm:
        os.remove(f'{estacao}.joblib')
    return model


# In[181]:


model = download_model('BAURU')


# In[182]:


from RainPredictionMachine.data import CleanDataRpm
cleaner = CleanDataRpm()
df = cleaner.clean_data(2, gcp=False)
# X_train, y_train, X_test, y_test = split_train_test(df,6000,72)


# In[183]:


pd.DataFrame(X_test[2])


# In[27]:


#X_train = X_train[:,:,:17]


# In[184]:


X_train.shape


# In[29]:


# X_test = X_test[:,:,:17]


# In[185]:


X_test.shape


# In[186]:


# 6000 sequencias para as proximas 24 horas
model.evaluate(X_test,y_test)


# In[187]:


# Avaliar nossa previsao
y_pred = model.predict(X_test)
y_pred


# In[102]:


# Transformar o y_test e y_pred em DataFrame
df_y_pred = pd.DataFrame(y_pred).T
df_y_test = pd.DataFrame(y_test).T


# In[103]:


def classe_chuva(precipitacao):
    mm=precipitacao
    if np.isnan(mm):
        chuva = "NaN"
    if mm == 0:
        chuva = 0 #'nao chove'
    elif mm >0 and mm <=5.0:
        chuva = 1 #'fraca'
    elif mm >5.0 and mm<=25.0:
        chuva = 2 #'moderada'
    else:
        chuva = 3 #'forte'
    return chuva


# In[104]:


# Baseado na nossa classe_chuva, definimos os elemento de acordo com a intensidade da precipitacao
df_y_test_categorical = df_y_test.applymap(classe_chuva)
df_y_test_categorical.T.value_counts()


# In[190]:


df_y_pred_categorical = df_y_pred.applymap(classe_chuva)
df_y_pred_categorical.T.value_counts()


# In[106]:


# Igualando para saber o quanto acertamos 
(df_y_test_categorical == df_y_pred_categorical).sum()


# In[107]:


(df_y_test_categorical == df_y_pred_categorical).sum().sum()


# In[108]:


(df_y_test_categorical == df_y_pred_categorical).sum().sum()/(6000*24) #accuracy  (n_sequences, length)


# In[48]:


(df_y_test_categorical == df_y_pred_categorical).sum(axis=1)/6000


# In[113]:


df_y_test_categorical.shape


# In[110]:


df_y_pred_categorical.shape


# In[198]:


numpy_y_pred = df_y_pred_categorical.to_numpy()[0,:]


# In[199]:


numpy_y_test = df_y_test_categorical.to_numpy()[0,:]


# In[58]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[201]:


classification_report(numpy_y_test, numpy_y_pred, output_dict=True)['0']


# In[250]:


import numpy as np
# precision_m, recall_m, f1_score_m, support_m = [], [], [], []
classification_reports = []
                                                

for i in range(0,24):
    numpy_y_pred = df_y_test_categorical.to_numpy()[i,:]
    numpy_y_test = df_y_pred_categorical.to_numpy()[i,:]
    dict_classificacao = classification_report(numpy_y_test, numpy_y_pred, output_dict=True)
    dicionario2 = {}
    for key in list(dict_classificacao.keys())[0:4]:
#         dicionario2 = dict_classificacao[key]
        for key2 in list(dict_classificacao[key].keys()):
            dicionario2[f'{key2}, {key}'] = dict_classificacao[key][key2]
#             dicionario2.pop(key2)
    classification_reports.append(dicionario2)


# In[251]:


dicionario2


# In[252]:


pd.DataFrame(classification_reports)


# In[253]:


numpy_y_pred = df_y_test_categorical.to_numpy()[i,:]
numpy_y_test = df_y_pred_categorical.to_numpy()[i,:]    


# In[254]:


pd.Series(numpy_y_test).value_counts() #previsao para 24horas


# In[255]:


pd.Series(numpy_y_pred).value_counts() #previsao para 24horas


# In[256]:


print(classification_report(numpy_y_test, numpy_y_pred, output_dict=True))


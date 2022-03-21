import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.impute import KNNImputer

class clean_data_rpm():
    def __init__(self):
        self.files = None #instanciar files que ainda nao foi definido
    #Criando variáveis organizacionais
    def get_data(self,n_arquivos):
        path = '../raw_data/SP' #caminho geral
        files = [f for f in listdir(path) if isfile(join(path, f))] #lista de nomes de arquivos de dados
        #Loop para fazer lista com os dataframes, ignorando o cabeçalho
        #Criando 4 novas features a partir de infos do cabeçalho
        df_list = []
        for file in range(0,n_arquivos):
            df = pd.read_csv(f'{path}/{files[file]}', sep=';', skiprows=8, encoding="ISO-8859-1", decimal=',')
            lat_log_alt = pd.read_csv(f'{path}/{files[file]}', sep=';', skiprows=4,
                            nrows=3, encoding="ISO-8859-1", decimal=',', names=['lat_lon_alt','valor'])
            df['Estaçao']=files[file].split('_')[4]
            df['Latitude']=lat_log_alt['valor'][0]
            df['Longitude']=lat_log_alt['valor'][1]
            df['Altitude']=lat_log_alt['valor'][2]
            df_list.append(df)
        return df_list

    def clean_data(self,n_arquivos):
        df_list = self.get_data(n_arquivos) #chamar função dentro de classe
        #fundir os dataframes no dataframe vazio
        full_df = pd.concat(df_list)
        df2 = full_df.copy()
        #dropar coluna inútil que foi criada por ter ; no final da linha do arquivo csv
        df2.drop(columns=["Unnamed: 19"],inplace=True)
        #mudar nome das colunas
        df2= df2.rename(columns={'Data': 'Data',
                                    'Hora UTC': 'Hora(UTC)',
                                    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'Chuva',
                                    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'Pres',
                                    'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)': 'Pres_max',
                                    'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)': 'Pres_min',
                                    'RADIACAO GLOBAL (Kj/m²)': 'Radiacao',
                                    'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'Temp',
                                    'TEMPERATURA DO PONTO DE ORVALHO (°C)': 'Temp_orvalho',
                                    'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'Temp_max',
                                    'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'Temp_min',
                                    'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)': 'Temp_orvalho_max',
                                    'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)': 'Temp_orvalho_min',
                                    'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)': 'Umid_max',
                                    'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)': 'Umid_min',
                                    'UMIDADE RELATIVA DO AR, HORARIA (%)': 'Umid',
                                    'VENTO, DIREÇÃO HORARIA (gr) (° (gr))': 'Dir_vento',
                                    'VENTO, RAJADA MAXIMA (m/s)': 'Rajada_vento',
                                    'VENTO, VELOCIDADE HORARIA (m/s)': 'Vel_vento'})
        #criar coluna que uni data e hora
        df2['datahora'] = df2.apply(lambda row: pd.to_datetime(f'{row["Data"]} {row["Hora(UTC)"]}'), axis=1)
        #dropar colunas 'Data' e 'Hora(UTC)'
        df2.drop(columns=["Data"], inplace=True)
        df2.drop(columns=["Hora(UTC)"], inplace=True)
        #Tratando valores nulos de radiação de noite
        df2['Radiacao'] = df2.apply(lambda x: self.tratar_radiacao(x['datahora'],x['Radiacao']),axis=1) #chamar função dentro de classe
        #Fazer imputer nas colunas --- precisa ser aqui pq a radiacao tem que ter trocado o nan da noite por 0
        imputer = KNNImputer(n_neighbors=2)
        colunas_imputer = ['Rajada_vento', 'Vel_vento', 'Dir_vento', 'Radiacao', 'Temp_orvalho_min', 'Temp_orvalho_max', 'Umid_min', 'Umid_max', 'Temp_orvalho', 'Umid', 'Pres_max', 'Pres_min', 'Temp_max', 'Temp_min', 'Chuva', 'Pres', 'Temp']
        df2[colunas_imputer] = imputer.fit_transform(df2[colunas_imputer])
        #Transformando chuva em variável categórica na coluna 'classe_chuva'
        df2['classe_chuva'] = df2['Chuva'].apply(lambda x: self.classe_chuva(x)) #chamar função dentro de classe
        return df2

    #---------------------transformando valores nan para 0 da radiação de noite---------------------
    def tratar_radiacao(self, hora,radiacao):
        if (hora.hour >= 22) or (hora.hour <= 8):
            if np.isnan(radiacao):
                radiacao = 0
        return radiacao

    #---------------------transformando a chuva em variável categórica---------------------
    def classe_chuva(self, precipitacao):
        mm=precipitacao
        if np.isnan(mm):
            chuva = "NaN"
        if mm == 0:
            chuva = 'nao chove'
        elif mm >0 and mm <=5.0:
            chuva = 'fraca'
        elif mm >5.0 and mm<=25.0:
            chuva = 'moderada'
        else:
            chuva = 'forte'
        return chuva

if __name__ == "__main__":
    instan_clean_data_rpm = clean_data_rpm() #instanciar a classe
    #Testes para ver se as duas funões estão funcionando
    #Open data
    print('abrindo os dados')
    instan_clean_data_rpm.get_data()
    #Clean data
    print('limpando os dados')
    instan_clean_data_rpm.clean_data()

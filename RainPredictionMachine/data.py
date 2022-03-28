import pandas as pd
from os import listdir
import os
from os.path import isfile, join
import numpy as np
from sklearn.impute import KNNImputer
from google.cloud import storage

class CleanDataRpm():

    def __init__(self):
        self.files = None #SELF PARA VARIAVEIS QUE VAMOS USAR DEPOIS
        self.client = storage.Client()
        self.pathh = os.path.join(os.path.dirname(os.path.dirname(__file__)),'raw_data','SP') #caminho geral
        self.files = [f for f in listdir(self.pathh) if isfile(join(self.pathh, f))] #lista de nomes de arquivos de dados
        self.cidades=['BARRETOS', 'BARUERI', 'BAURU', 'BEBEDOURO', 'BERTIOGA',
        'BRAGANCA PAULISTA', 'CACHOEIRA PAULISTA', 'CAMPOS DO JORDAO',
        'CASA BRANCA', 'DRACENA', 'FRANCA', 'IGUAPE', 'ITAPEVA', 'ITAPIRA',
        'ITATIAIA', 'ITUVERAVA', 'LINS', 'MARILIA', 'OURINHOS', 'PARATI',
        'PIRACICABA', 'PRADOPOLIS', 'PRESIDENTE PRUDENTE', 'RANCHARIA',
        'SAO CARLOS', 'SAO LUIS DO PARAITINGA', 'SAO MIGUEL ARCANJO',
        'SAO PAULO - INTERLAGOS', 'SAO PAULO - MIRANTE', 'SAO SEBASTIAO',
        'SAO SIMAO', 'SOROCABA', 'TAUBATE', 'TUPA', 'VALPARAISO', 'VOTUPORANGA']
    #Criando variáveis organizacionais

    def get_data(self,n_cidade):#escolher um numero de 0 a 35
        #Loop para fazer lista com os dataframes, ignorando o cabeçalho
        #Criando 4 novas features a partir de infos do cabeçalho
        file = []
        cidade = self.cidades[n_cidade]
        for i in range(0, 63):#aqui eu seleciono somente os arquivos com barreto no nome
            if cidade in self.files[i]:
                file.append(self.files[i])

        df_list = [] #aqui eu crio dataframe soh com a cidade selecionada

        for ii in range(0, len(file)):
            df = pd.read_csv(f'{self.pathh}/{file[ii]}', sep=';', skiprows=8, encoding="ISO-8859-1", decimal=',')
            lat_log_alt = pd.read_csv(f'{self.pathh}/{file[ii]}', sep=';', skiprows=4,
                            nrows=3, encoding="ISO-8859-1", decimal=',', names=['lat_lon_alt','valor'])
            # df['Estaçao']=file[ii].split('_')[4]
            df['Latitude']=lat_log_alt['valor'][0]
            df['Longitude']=lat_log_alt['valor'][1]
            df['Altitude']=lat_log_alt['valor'][2]
            df_list.append(df)

        return df_list

#ve se essa parte esta certa
    def get_gcp_data(self, n_cidade): #get_gcp_data para rodar no colab
        bucket = self.client.get_bucket('rain-prediction-machine') #
        files = [str(blob.name) for blob in bucket.list_blobs()] #blob = arquivo bucket = diretorio
        file = []
        cidade = self.cidades[n_cidade]
        for i in range(0, 63):#aqui eu seleciono somente os arquivos com barreto no nome
            if cidade in files[i]:
                file.append(files[i])

        df_list = [] #aqui eu crio dataframe soh com a cidade selecionada

        for ii in range(0, len(file)):
            df = pd.read_csv(f'gs://rain-prediction-machine/{file[ii]}', sep=';', skiprows=8, encoding="ISO-8859-1", decimal=',')
            lat_log_alt = pd.read_csv(f'gs://rain-prediction-machine/{file[ii]}', sep=';', skiprows=4,
                            nrows=3, encoding="ISO-8859-1", decimal=',', names=['lat_lon_alt','valor'])
            # df['Estaçao']=files[file].split('_')[4]
            df['Latitude']=lat_log_alt['valor'][0]
            df['Longitude']=lat_log_alt['valor'][1]
            df['Altitude']=lat_log_alt['valor'][2]
            df_list.append(df)
        return df_list

    def clean_data(self, n_cidade, gcp=False):
        if gcp:
            df_list = self.get_gcp_data(n_cidade)
        else:
            df_list = self.get_data(n_cidade) #chamar função dentro de classe
        #fundir os dataframes no dataframe vazio
        if len(df_list) > 1:
            full_df = pd.concat(df_list)
        else:
            full_df = df_list[0]
        df2 = full_df.copy()
        #dropar coluna inútil que foi criada por ter ; no final da linha do arquivo csv
        df2.drop(columns=["Unnamed: 19"],inplace=True)

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
        # dropar coluna Chuva
        # df2.drop(columns=["Chuva"],inplace=True)
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
            chuva = 0 #'nao chove'
        elif mm >0 and mm <=5.0:
            chuva = 1 #'fraca'
        elif mm >5.0 and mm<=25.0:
            chuva = 2 #'moderada'
        else:
            chuva = 3 #'forte'
        return chuva

    def get_lat_lon(self, n_files):
        estacao, lat, lon = [], [], []
        for file in range(0,n_files):
            lat_lon_alt = pd.read_csv(f'{self.pathh}/{self.files[file]}', sep=';', skiprows=4,
                             nrows=3, encoding="ISO-8859-1", decimal=',', names=['lat_lon_alt','valor'])
            est=self.files[file].split('_')[4]
            latt=lat_lon_alt['valor'][0]
            lonn=lat_lon_alt['valor'][1]
            estacao.append(est)
            lat.append(latt)
            lon.append(lonn)
        return lat, lon, estacao

if __name__ == "__main__":

    instan_clean_data_rpm = CleanDataRpm() #instanciar a classe
    #Testes para ver se as duas funões estão funcionando
    #Open data
    #print('abrindo os dados')
    #instan_clean_data_rpm.get_data()
    #Clean data
    #print('limpando os dados')
    #instan_clean_data_rpm.clean_data()
    df = instan_clean_data_rpm.clean_data(0, gcp=False)
    print(len(df))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pytz
import requests
import datetime
import joblib
from google.cloud import storage
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins permite qlqr um acessar
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods permite qlqr tipo de metodo para acessar
    allow_headers=["*"],  # Allows all headers
)
#Função para fazer o dumping do predict no google cloud, criar diretorio front para receber os arquivos predict e passado.csv
def upload_csv_to_gcp(file):
        client = storage.Client()
        bucket = client.bucket('rain-prediction-machine')
        blob = bucket.blob(f'front/{file}')
        blob.upload_from_filename(file)

@app.get("/")
def index():
    return {"pickup_datetime": "2013-07-06 17:18:00",
            "pickup_longitude": "-73.950655",
            "pickup_latitude": "40.783282",
            "dropoff_longitude": "-73.984365",
            "dropoff_latitude": "40.769802",
            "passenger_count": "1"}
            #{"greeting": "Hello world"}

@app.get("/predict")

def predict():
    #ajustar csv do google cloud
    read_csv_to_api = pd.read_csv('gs://rain-prediction-machine/front/info_to_api.csv')
    codigosestacao = read_csv_to_api.CodigoEstacao.to_list()
    nome_modelos = read_csv_to_api.Estacao.apply(lambda name: name.split(' ')[-1]).to_list()
    #codigosestacao = ['A748']
    #nome_modelos = ['BARRETOS']
    
    lista_df = []
    lista_df_passado = []
    for codigoestacao, nome_modelo in zip(codigosestacao, nome_modelos):
        print(nome_modelo)
        #Pegando a  hora atual e diminuindo 3 antes
        hora_now = datetime.datetime.now()
        dia_atual = hora_now.strftime("%Y-%m-%d")
        dia_pas = (hora_now - datetime.timedelta(days=3)).strftime("%Y-%m-%d")

        #Request da API do INMET
        url = f'https://apitempo.inmet.gov.br/estacao/{dia_pas}/{dia_atual}/{codigoestacao}'
        response = requests.get(url).json()

        #Tratando as informações para prever com nosso modelo
        df = pd.DataFrame(response)
        dc_nome = df['DC_NOME'][0]
        df.drop(columns=['DC_NOME', 'UF', 'HR_MEDICAO', 'CD_ESTACAO', 'DT_MEDICAO', 'TEM_SEN'], inplace=True)
        df.dropna(inplace=True)
        if df.shape[0] < 72:
            print(f'tirar {nome_modelo}')
            continue
        df = df.iloc[-48:].reset_index(drop=True)
        df = df[['CHUVA', 'PRE_INS', 'PRE_MAX', 'PRE_MIN', 'RAD_GLO', 'TEM_INS', 'PTO_INS', 'TEM_MAX', 'TEM_MIN', 'PTO_MAX', 'PTO_MIN', 'UMD_MAX', 'UMD_MIN', 'UMD_INS', 'VEN_DIR', 'VEN_RAJ', 'VEN_VEL', 'VL_LATITUDE', 'VL_LONGITUDE']]
        df = df.rename(columns={'CHUVA': 'Chuva',
                                        'PRE_INS': 'Pres',
                                        'PRE_MAX': 'Pres_max',
                                        'PRE_MIN': 'Pres_min',
                                        'RAD_GLO': 'Radiacao',
                                        'TEM_INS': 'Temp',
                                        'PTO_INS': 'Temp_orvalho',
                                        'TEM_MAX': 'Temp_max',
                                        'TEM_MIN': 'Temp_min',
                                        'PTO_MAX': 'Temp_orvalho_max',
                                        'PTO_MIN': 'Temp_orvalho_min',
                                        'UMD_MAX': 'Umid_max',
                                        'UMD_MIN': 'Umid_min',
                                        'UMD_INS': 'Umid',
                                        'VEN_DIR': 'Dir_vento',
                                        'VEN_RAJ': 'Rajada_vento',
                                        'VEN_VEL': 'Vel_vento',
                                        'VL_LATITUDE': 'Latitude',
                                        'VL_LONGITUDE': 'Longitude'})
        #ajustar caminho para pegar os dados de maneira generica
        #alti = read_csv_to_api[read_csv_to_api['CodigoEstacao'] == codigoestacao]['Altitude'].values[0]
        #Adicionando coluna que nao veio da API do INMET
       # df['Altitude'] = alti
        df = df.astype(float)
        latitude = df['Latitude']
        longitude = df['Longitude']
        df = df.drop(columns = ['Latitude', 'Longitude'])
        X_test = pd.DataFrame(df).to_numpy().reshape(1,48,17)
        gcs_path = f'gs://rain-prediction-machine/models/{nome_modelo}.joblib'
        model = joblib.load(tf.io.gfile.GFile(gcs_path, 'rb')) #para ler um joblib precisa da ajuda do tensor flow
        y_pred = model.predict(X_test)
        # Ajustando o df para ler no front end
        df_pred = pd.DataFrame(y_pred)
        df_pred['dc_nome'] = dc_nome
        df_pred['Latitude'] = latitude
        df_pred['Longitude'] = longitude
        df['dc_nome'] = dc_nome
        lista_df.append(df_pred)
        lista_df_passado.append(df)
    pred_all_esta = pd.concat(lista_df)
    pred_all_esta.to_csv('exemplo_nat_all.csv')
    upload_csv_to_gcp('exemplo_nat_all.csv')
    pd.concat(lista_df_passado).to_csv('df_passad.csv')
    upload_csv_to_gcp('df_passad.csv')
    #concatenar todos os df pred
    #enviar o df pred concatenado para google cloud
    #enviar df com velocidade do vento, umidade e temp max e temp min atualizado para a Nat
    return {"Predict": pred_all_esta.to_dict(), "Passado": lista_df_passado}

    
#Fazer endpoint para ler no bucket
@app.get("/bucket")
def read():
    prev = pd.read_csv('gs://rain-prediction-machine/front/exemplo_nat_all.csv')
    passa = pd.read_csv('gs://rain-prediction-machine/front/df_passad.csv')
    return {'Previsao': prev.to_dict(), 'Passado': passa.to_dict()} #api so le dict e lista
    
#Deixar API online para desenvolvedores


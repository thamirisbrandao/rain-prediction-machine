import streamlit as st
import requests
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from RainPredictionMachine.data import CleanDataRpm
'''
# Rain Prediction Machine
'''

st.markdown('''
Tentando criar um website para previsão de chuva''')

option = st.selectbox(
    'Escolha uma estação meteorológica',
    ('BARRETOS', 'BARUERI', 'BAURU', 'BEBEDOURO', 'BERTIOGA',
     'BRAGANCA PAULISTA', 'CACHOEIRA PAULISTA', 'CAMPOS DO JORDAO',
     'CASA BRANCA', 'DRACENA', 'FRANCA', 'IGUAPE', 'ITAPEVA', 'ITAPIRA',
     'ITATIAIA', 'ITUVERAVA', 'LINS', 'MARILIA', 'OURINHOS', 'PARATI',
     'PIRACICABA', 'PRADOPOLIS', 'PRESIDENTE PRUDENTE', 'RANCHARIA',
     'SAO CARLOS', 'SAO LUIS DO PARAITINGA', 'SAO MIGUEL ARCANJO',
     'SAO PAULO - INTERLAGOS', 'SAO PAULO - MIRANTE', 'SAO SEBASTIAO',
     'SAO SIMAO', 'SOROCABA', 'TAUBATE', 'TUPA', 'VALPARAISO', 'VOTUPORANGA'))


st.write('You selected:', option)


# def get_lat_lon(n_files):
#     pathh = 'raw_data/SP'  #caminho geral
#     files = [f for f in listdir(pathh) if isfile(join(pathh, f))]
#     estacao, lat, lon = [], [], []

#     for file in range(0, n_files):
#         lat_lon_alt = pd.read_csv(f'raw_data/SP/{files[file]}',
#                                   sep=';',
#                                   skiprows=4,
#                                   nrows=3,
#                                   encoding="ISO-8859-1",
#                                   decimal=',',
#                                   names=['lat_lon_alt', 'valor'])
#         est = files[file].split('_')[4]
#         latt = lat_lon_alt['valor'][0]
#         lonn = lat_lon_alt['valor'][1]
#         estacao.append(est)
#         lat.append(latt)
#         lon.append(lonn)

#     return lat, lon, estacao

data=CleanDataRpm()

lat, lon, estacao = data.get_lat_lon(63)
coord=pd.DataFrame(lat, columns=['lat'])
coord['lon']=lon
coord['estacao']=estacao
coord.drop_duplicates('estacao')
coord.drop(columns='estacao')

# df = pd.DataFrame(
#      np.random.randn(100, 2) / [-24.67166666, -20.35972222] + [-51.552254, -44.70305555],
#      columns=['lat', 'lon'])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperatura", "23 °C", "1.2 °C")
col2.metric("Vento", "9 m/s", "-8%")
col3.metric("Umidade", "86%", "4%")
col4.metric("Chuva", "mm", "4 mm")

st.map(coord)

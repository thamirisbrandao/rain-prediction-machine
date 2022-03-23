import streamlit as st
import requests
import pandas as pd
import numpy as np
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
       'SAO SIMAO', 'SOROCABA', 'TAUBATE', 'TUPA', 'VALPARAISO',
       'VOTUPORANGA'))

st.write('You selected:', option)

data=CleanDataRpm()
lat, lon, estacao =data.get_lat_lon(63)
coord=pd.DataFrame([lat, lon, estacao], columns=['lat'])
coord['lon']=lon
coord['estacao']=estacao
coord.drop_duplicates('estacao')

# df = pd.DataFrame(
#      np.random.randn(100, 2) / [-24.67166666, -20.35972222] + [-51.552254, -44.70305555],
#      columns=['lat', 'lon'])

st.map(coord)

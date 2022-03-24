import streamlit as st
import requests
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from RainPredictionMachine.data import CleanDataRpm
import pydeck as pdk
import altair as alt

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

data=CleanDataRpm()
'''

'''
n_cidade=data.cidades.index(option)
st.write('view',n_cidade)
df=data.clean_data(n_cidade)
st.write('temp',df.Temp.iloc[-1])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperatura", f'{df.Temp.iloc[-1]} °C',
            f'{round(df.Temp.iloc[-1]-df.Temp.iloc[-24],ndigits=1)} °C')
col2.metric("Vento", f'{df.Vel_vento.iloc[-1]} m/s',
        f'{round(df.Vel_vento.iloc[-1]-df.Vel_vento.iloc[-24],ndigits=1)} m/s')
col3.metric("Umidade", f'{df.Umid.iloc[-1]} %',
        f'{round(df.Umid.iloc[-1]-df.Umid.iloc[-24],ndigits=1)} %')
col4.metric("Chuva", f'{df.Chuva.iloc[-1]} mm',
        f'{round(df.Chuva.iloc[-1]-df.Chuva.iloc[-24],ndigits=1)} mm')

lat, lon, estacao = data.get_lat_lon(63)
coord=pd.DataFrame(lat, columns=['lat'])
coord['lon']=lon
coord['estacao']=estacao
coord.drop_duplicates('estacao')
coord.drop(columns='estacao')

'''

'''
st.map(coord)

''

''



st.line_chart(df.Temp.iloc[-72:])
st.line_chart(df.Vel_vento.iloc[-72:])

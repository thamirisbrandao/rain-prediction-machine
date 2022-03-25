from curses.panel import bottom_panel
from tkinter import W
import streamlit as st
import requests
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from RainPredictionMachine.data import CleanDataRpm
import pydeck as pdk
import datetime
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Alerta Chuva SP')

st.write("### Atenção :warning:")

st.write(f"Chuva forte ou muito forte paras as seguintes cidades: ")

st.header('Previsão de chuva para as próximas 24h')

'''

'''

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


dia0 =datetime.date.today()#dia corrente
delta = datetime.timedelta(days=1)#espaço de tempo
dia1=dia0+delta#dia para mais 24h

d1 = st.date_input("Previsão para:", dia0, max_value = dia1, min_value=dia0)
t1 = st.time_input('Hora', datetime.time(0, 0))

'''

'''
data=CleanDataRpm()
n_cidade=data.cidades.index(option)
# st.write('view',n_cidade)
df=data.clean_data(n_cidade)
# st.write('temp',df.Temp.iloc[-1])

# st.write(f"#### {option} {d1}")
st.write('#### Previsão do tempo')

if df.Chuva.iloc[-1] == 0:
    st.write(f'#### Chuva:    {data.classe_chuva(df.Chuva.iloc[-1])}.')
if df.Chuva.iloc[-1] > 0 and df.Chuva.iloc[-1] < 25:
    st.write(f'#### Chuva:    {data.classe_chuva(df.Chuva.iloc[-1])} ☔')
else:
    st.write(f'#### Chuva:    {data.classe_chuva(df.Chuva.iloc[-1])} ⛈️')


col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperatura", f'{df.Temp.iloc[-1]} °C',
            f'{round(df.Temp.iloc[-1]-df.Temp.iloc[-24],ndigits=1)} °C')
col2.metric("Vento", f'{round(df.Vel_vento.iloc[-1],ndigits=1)} m/s',
        f'{round(df.Vel_vento.iloc[-1]-df.Vel_vento.iloc[-24],ndigits=1)} m/s')
col3.metric("Umidade", f'{df.Umid.iloc[-1]} %',
        f'{round(df.Umid.iloc[-1]-df.Umid.iloc[-24],ndigits=1)} %')
col4.metric("Precipitação", f'{df.Chuva.iloc[-1]} mm',
        f'{round(df.Chuva.iloc[-1]-df.Chuva.iloc[-24],ndigits=1)} mm')


# Chuva_ultimahora=[]
#mudar para saida do model, por enqt lendo o ultimo dado de cada arquivo

# for i in range(0,2):
#     df= data.clean_data(i)
#     Chuva_ultimahora.append(df.Chuva.iloc[-1])

Chuva_ultimahora=[30, 20, 25, 0, 10, 60, 0, 0, 4]

lat, lon, estacao = data.get_lat_lon(9)
coord=pd.DataFrame(lat, columns=['lat'])
coord['lon']=lon
coord['estacao']=estacao
coord.drop_duplicates('estacao')
coord['Chuva']=Chuva_ultimahora

#definindo colormap por quantidade de chuva
r,b,g = [],[],[]
for mm in Chuva_ultimahora:

    if mm == 0:
        cor = [0, 0, 0]
    elif mm >0 and mm <=5.0:
        cor = [128, 255, 0]
    elif mm >5.0 and mm<=25.0:
        cor = [0, 191, 255]
    elif mm >25.0 and mm<=50.0:
        cor = [191, 0, 255]
    else:
        cor = [255, 0, 0]
    r.append(cor[0])
    b.append(cor[1])
    g.append(cor[2])

coord['r']=r
coord['b']=b
coord['g']=g

'''

'''

# Define a layer to display on a map
layer = pdk.Layer(
    'ScatterplotLayer',
    coord,
    pickable=False,
    opacity=0.3,
    stroked=True,
    filled=True,
    radius_scale=1000,
    radius_min_pixels=5,
    radius_max_pixels=100,
    line_width_min_pixels=1,
    get_position=['lon','lat'],
    get_radius='Chuva',
    get_fill_color=['r','b','g'],
    get_line_color=['r','b','g']
)

'''



'''

# Set the viewport location
view_state = pdk.ViewState(latitude=-21.980353, longitude=-47.883927, zoom=5)
st.subheader(f'Previsão de Chuva para SP em {dia0}')
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state,
                         map_style="mapbox://styles/mapbox/light-v10", tooltip=coord['Chuva']))

# st.map(coord)
'''



'''
def plot_temp(days,min_t,max_t):
        fig = plt.figure(figsize=(20,8))
        plt.plot(days[-24*7:],max_t[-24*7:],color='green',linestyle='dashdot',linewidth = 1,
                 marker='o',markerfacecolor='red',markersize=7)
        plt.plot(days[-24*7:],min_t[-24*7:],color='orange',linestyle='dashdot',linewidth = 1,marker='o',markerfacecolor='blue',markersize=7)
        plt.ylim(min(min_t)-2,max(max_t)+2)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True,color='brown')
        plt.legend(["Temperatura Máxima","Temperatura Mínima"],loc=0, fontsize=20)
        # plt.xlabel('Data(mm/dd)')
        plt.ylabel('Temperatura °C', fontsize=25)
        # plt.title('6-Day Weather Forecast')
        st.pyplot(fig)
        # return fig

st.subheader('Temperatura do ar (°C) nos útimos sete dias')
plot_temp(df.datahora,df.Temp_min,df.Temp_max)

'''


'''
def plot_vento(days,vel_vento):
        fig = plt.figure(figsize=(20,8))
        plt.plot(days[-24*7:],vel_vento[-24*7:],color='blue',linestyle='-',linewidth = 1,
                 marker='o',markerfacecolor='blue',markersize=7)
        plt.ylim(0, max(vel_vento)+1)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True,color='brown')
        plt.legend(["Intensidade do vento"],loc=0, fontsize=20)
        # plt.xlabel('Data(mm/dd)')
        plt.ylabel('Vento m/s', fontsize=25)
        # plt.title('6-Day Weather Forecast')
        st.pyplot(fig)
        # return fig

st.subheader('Intensidade do vento (m/s) nos útimos sete dias')
plot_vento(df.datahora,df.Vel_vento)

def plot_umi(days,umidade):
        fig = plt.figure(figsize=(20,8))
        plt.plot(days[-24*7:],umidade[-24*7:],color='blue',linestyle='-',linewidth = 1,
                 marker='o',markerfacecolor='blue',markersize=7)
        plt.ylim(0, 110)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True,color='brown')
        plt.legend(["Umidade Relativa %"],loc=0, fontsize=20)
        # plt.xlabel('Data(mm/dd)')
        plt.ylabel('Umidade Relativa %', fontsize=25)
        # plt.title('6-Day Weather Forecast')
        st.pyplot(fig)
        # return fig

st.subheader('Umidade Relativa % nos útimos sete dias')
plot_umi(df.datahora,df.Umid)


# st.line_chart(df.Temp.iloc[-72:])
# st.line_chart(df.Vel_vento.iloc[-72:])

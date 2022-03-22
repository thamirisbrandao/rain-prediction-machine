{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "94eb2ee2",
   "metadata": {},
   "source": [
    "# Importações"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "cf612922",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from os import listdir\n",
    "from os.path import isfile, join\n",
    "import numpy as np\n",
    "from sklearn.impute import KNNImputer"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ff13f3c3",
   "metadata": {},
   "source": [
    "# Criando variáveis organizacionais"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a1e68af9",
   "metadata": {},
   "outputs": [],
   "source": [
    "path = '../raw_data/SP' #caminho geral\n",
    "files = [f for f in listdir(path) if isfile(join(path, f))] #lista de nomes de arquivos de dados"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dc269e2c",
   "metadata": {},
   "source": [
    "# Tratamentos iniciais dos dados"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2c3a140d",
   "metadata": {},
   "source": [
    "## Tratamentos básicos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "649ce8fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "path = '../raw_data/SP' #caminho geral\n",
    "files = [f for f in listdir(path) if isfile(join(path, f))] #lista de nomes de arquivos de dados\n",
    "#Loop para fazer lista com os dataframes, ignorando o cabeçalho \n",
    "#Criando 4 novas features a partir de infos do cabeçalho\n",
    "df_list = []\n",
    "for file in range(0,2):\n",
    "    df = pd.read_csv(f'../raw_data/SP/{files[file]}', sep=';', skiprows=8, encoding=\"ISO-8859-1\", decimal=',')\n",
    "    lat_log_alt = pd.read_csv(f'../raw_data/SP/{files[file]}', sep=';', skiprows=4,\n",
    "                    nrows=3, encoding=\"ISO-8859-1\", decimal=',', names=['lat_lon_alt','valor'])\n",
    "    df['Estaçao']=files[file].split('_')[4]\n",
    "    df['Latitude']=lat_log_alt['valor'][0]\n",
    "    df['Longitude']=lat_log_alt['valor'][1]\n",
    "    df['Altitude']=lat_log_alt['valor'][2]\n",
    "    df_list.append(df)\n",
    "\n",
    "#fundir os dataframes no dataframe vazio\n",
    "full_df = pd.concat(df_list)\n",
    "df2 = full_df.copy()\n",
    "#dropar coluna inútil que foi criada por ter ; no final da linha do arquivo csv\n",
    "df2.drop(columns=[\"Unnamed: 19\"],inplace=True)\n",
    "#mudar nome das colunas\n",
    "df2= df2.rename(columns={'Data': 'Data',\n",
    "                            'Hora UTC': 'Hora(UTC)',\n",
    "                            'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'Chuva',\n",
    "                            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'Pres',\n",
    "                            'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)': 'Pres_max',\n",
    "                            'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)': 'Pres_min',\n",
    "                            'RADIACAO GLOBAL (Kj/m²)': 'Radiacao',\n",
    "                            'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'Temp',\n",
    "                            'TEMPERATURA DO PONTO DE ORVALHO (°C)': 'Temp_orvalho',\n",
    "                            'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'Temp_max',\n",
    "                            'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'Temp_min',\n",
    "                            'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)': 'Temp_orvalho_max',\n",
    "                            'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)': 'Temp_orvalho_min',\n",
    "                            'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)': 'Umid_max',\n",
    "                            'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)': 'Umid_min',\n",
    "                            'UMIDADE RELATIVA DO AR, HORARIA (%)': 'Umid',\n",
    "                            'VENTO, DIREÇÃO HORARIA (gr) (° (gr))': 'Dir_vento',\n",
    "                            'VENTO, RAJADA MAXIMA (m/s)': 'Rajada_vento',\n",
    "                            'VENTO, VELOCIDADE HORARIA (m/s)': 'Vel_vento'})"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b5c86185",
   "metadata": {},
   "source": [
    "## Transformação de dados e preenchimento"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bcaec815",
   "metadata": {},
   "source": [
    "### Formatando data e hora"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1fc86e43",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#---------------------colocando data e hora no formato adequado---------------------\n",
    "#df2[\"Data\"] = pd.to_datetime(df2[\"Data\"])\n",
    "df2['datahora'] = df2.apply(lambda row: pd.to_datetime(f'{row[\"Data\"]} {row[\"Hora(UTC)\"]}'), axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "ebb05917",
   "metadata": {},
   "outputs": [],
   "source": [
    "#dropar colunas 'Data' e 'Hora(UTC)'\n",
    "df2.drop(columns=[\"Data\"], inplace=True)\n",
    "df2.drop(columns=[\"Hora(UTC)\"], inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b539269c",
   "metadata": {},
   "source": [
    "### Tratando valores nulos de radiação"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "dc328712",
   "metadata": {},
   "outputs": [],
   "source": [
    "def tratar_radiacao(hora,radiacao):\n",
    "    if (hora.hour >= 22) or (hora.hour <= 8):\n",
    "        if np.isnan(radiacao):\n",
    "            radiacao = 0\n",
    "    return radiacao\n",
    "    \n",
    "#Tratando valores nulos de radiação de noite \n",
    "df2['Radiacao'] = df2.apply(lambda x: tratar_radiacao(x['datahora'],x['Radiacao']),axis=1) #chamar função dentro de classe"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cef579e5",
   "metadata": {},
   "source": [
    "### Imputer da variável de vento"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a16ff979",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#Fazer imputer nas colunas --- precisa ser aqui pq a radiacao tem que ter trocado o nan da noite por 0\n",
    "imputer = KNNImputer(n_neighbors=2)\n",
    "colunas_imputer = ['Rajada_vento', 'Vel_vento', 'Dir_vento', 'Radiacao', 'Temp_orvalho_min', 'Temp_orvalho_max', 'Umid_min', 'Umid_max', 'Temp_orvalho', 'Umid', 'Pres_max', 'Pres_min', 'Temp_max', 'Temp_min', 'Chuva', 'Pres', 'Temp']\n",
    "df2[colunas_imputer] = imputer.fit_transform(df2[colunas_imputer])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1a930a7f",
   "metadata": {},
   "source": [
    "### Transformando chuva em variável categórica"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b2f10bc1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#---------------------transformando a chuva em variável categórica---------------------\n",
    "def classe_chuva(precipitacao):\n",
    "    mm=precipitacao\n",
    "    if np.isnan(mm):\n",
    "        chuva = \"NaN\"\n",
    "    if mm == 0:\n",
    "        chuva = 'nao chove'\n",
    "    elif mm >0 and mm <=5.0:\n",
    "        chuva = 'fraca'\n",
    "    elif mm >5.0 and mm<=25.0:\n",
    "        chuva = 'moderada'\n",
    "    elif mm >25.0 and mm<=50:\n",
    "        chuva = 'forte'\n",
    "    else:\n",
    "        chuva = 'muito forte'\n",
    "    return chuva\n",
    "#Transformando chuva em variável categórica na coluna 'classe_chuva'\n",
    "df2['classe_chuva'] = df2['Chuva'].apply(lambda x: classe_chuva(x)) #chamar função dentro de classe"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f0475f48",
   "metadata": {},
   "source": [
    "# Explorando os dados"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "110e8de3",
   "metadata": {},
   "source": [
    "## Verificando células nulas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "c43dbb72",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Chuva               0.0\n",
       "Umid_min            0.0\n",
       "datahora            0.0\n",
       "Altitude            0.0\n",
       "Longitude           0.0\n",
       "Latitude            0.0\n",
       "Estaçao             0.0\n",
       "Vel_vento           0.0\n",
       "Rajada_vento        0.0\n",
       "Dir_vento           0.0\n",
       "Umid                0.0\n",
       "Umid_max            0.0\n",
       "Pres                0.0\n",
       "Temp_orvalho_min    0.0\n",
       "Temp_orvalho_max    0.0\n",
       "Temp_min            0.0\n",
       "Temp_max            0.0\n",
       "Temp_orvalho        0.0\n",
       "Temp                0.0\n",
       "Radiacao            0.0\n",
       "Pres_min            0.0\n",
       "Pres_max            0.0\n",
       "classe_chuva        0.0\n",
       "dtype: float64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2.isnull().sum().sort_values(ascending=False)/len(df2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "04d9f826",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Chuva</th>\n",
       "      <th>Pres</th>\n",
       "      <th>Pres_max</th>\n",
       "      <th>Pres_min</th>\n",
       "      <th>Radiacao</th>\n",
       "      <th>Temp</th>\n",
       "      <th>Temp_orvalho</th>\n",
       "      <th>Temp_max</th>\n",
       "      <th>Temp_min</th>\n",
       "      <th>Temp_orvalho_max</th>\n",
       "      <th>...</th>\n",
       "      <th>Umid</th>\n",
       "      <th>Dir_vento</th>\n",
       "      <th>Rajada_vento</th>\n",
       "      <th>Vel_vento</th>\n",
       "      <th>Estaçao</th>\n",
       "      <th>Latitude</th>\n",
       "      <th>Longitude</th>\n",
       "      <th>Altitude</th>\n",
       "      <th>datahora</th>\n",
       "      <th>classe_chuva</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>923.1</td>\n",
       "      <td>923.1</td>\n",
       "      <td>922.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>20.5</td>\n",
       "      <td>22.2</td>\n",
       "      <td>21.8</td>\n",
       "      <td>20.5</td>\n",
       "      <td>...</td>\n",
       "      <td>92.0</td>\n",
       "      <td>172.0</td>\n",
       "      <td>2.8</td>\n",
       "      <td>0.4</td>\n",
       "      <td>SAO PAULO - INTERLAGOS</td>\n",
       "      <td>-23.724501</td>\n",
       "      <td>-46.677501</td>\n",
       "      <td>771.00</td>\n",
       "      <td>2020-01-01 00:00:00+00:00</td>\n",
       "      <td>nao chove</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>923.7</td>\n",
       "      <td>923.7</td>\n",
       "      <td>923.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>22.1</td>\n",
       "      <td>20.7</td>\n",
       "      <td>22.3</td>\n",
       "      <td>22.0</td>\n",
       "      <td>20.8</td>\n",
       "      <td>...</td>\n",
       "      <td>92.0</td>\n",
       "      <td>101.0</td>\n",
       "      <td>1.8</td>\n",
       "      <td>0.9</td>\n",
       "      <td>SAO PAULO - INTERLAGOS</td>\n",
       "      <td>-23.724501</td>\n",
       "      <td>-46.677501</td>\n",
       "      <td>771.00</td>\n",
       "      <td>2020-01-01 01:00:00+00:00</td>\n",
       "      <td>nao chove</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0</td>\n",
       "      <td>923.6</td>\n",
       "      <td>923.8</td>\n",
       "      <td>923.6</td>\n",
       "      <td>0.0</td>\n",
       "      <td>23.2</td>\n",
       "      <td>20.1</td>\n",
       "      <td>23.5</td>\n",
       "      <td>22.1</td>\n",
       "      <td>20.8</td>\n",
       "      <td>...</td>\n",
       "      <td>82.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>3.3</td>\n",
       "      <td>1.2</td>\n",
       "      <td>SAO PAULO - INTERLAGOS</td>\n",
       "      <td>-23.724501</td>\n",
       "      <td>-46.677501</td>\n",
       "      <td>771.00</td>\n",
       "      <td>2020-01-01 02:00:00+00:00</td>\n",
       "      <td>nao chove</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0</td>\n",
       "      <td>923.1</td>\n",
       "      <td>923.6</td>\n",
       "      <td>923.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>23.3</td>\n",
       "      <td>19.5</td>\n",
       "      <td>23.6</td>\n",
       "      <td>23.1</td>\n",
       "      <td>20.1</td>\n",
       "      <td>...</td>\n",
       "      <td>79.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.2</td>\n",
       "      <td>SAO PAULO - INTERLAGOS</td>\n",
       "      <td>-23.724501</td>\n",
       "      <td>-46.677501</td>\n",
       "      <td>771.00</td>\n",
       "      <td>2020-01-01 03:00:00+00:00</td>\n",
       "      <td>nao chove</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>922.6</td>\n",
       "      <td>923.1</td>\n",
       "      <td>922.6</td>\n",
       "      <td>0.0</td>\n",
       "      <td>22.9</td>\n",
       "      <td>19.0</td>\n",
       "      <td>23.6</td>\n",
       "      <td>22.8</td>\n",
       "      <td>19.5</td>\n",
       "      <td>...</td>\n",
       "      <td>79.0</td>\n",
       "      <td>345.0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>0.6</td>\n",
       "      <td>SAO PAULO - INTERLAGOS</td>\n",
       "      <td>-23.724501</td>\n",
       "      <td>-46.677501</td>\n",
       "      <td>771.00</td>\n",
       "      <td>2020-01-01 04:00:00+00:00</td>\n",
       "      <td>nao chove</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8779</th>\n",
       "      <td>0.0</td>\n",
       "      <td>965.8</td>\n",
       "      <td>966.0</td>\n",
       "      <td>965.6</td>\n",
       "      <td>1636.0</td>\n",
       "      <td>26.7</td>\n",
       "      <td>22.2</td>\n",
       "      <td>27.6</td>\n",
       "      <td>26.2</td>\n",
       "      <td>23.3</td>\n",
       "      <td>...</td>\n",
       "      <td>76.0</td>\n",
       "      <td>310.0</td>\n",
       "      <td>6.3</td>\n",
       "      <td>2.4</td>\n",
       "      <td>RANCHARIA</td>\n",
       "      <td>-22.372832</td>\n",
       "      <td>-50.974710</td>\n",
       "      <td>398.75</td>\n",
       "      <td>2020-12-31 19:00:00+00:00</td>\n",
       "      <td>nao chove</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8780</th>\n",
       "      <td>0.0</td>\n",
       "      <td>966.2</td>\n",
       "      <td>966.2</td>\n",
       "      <td>965.8</td>\n",
       "      <td>751.1</td>\n",
       "      <td>25.7</td>\n",
       "      <td>21.8</td>\n",
       "      <td>27.0</td>\n",
       "      <td>25.7</td>\n",
       "      <td>22.8</td>\n",
       "      <td>...</td>\n",
       "      <td>79.0</td>\n",
       "      <td>308.0</td>\n",
       "      <td>5.5</td>\n",
       "      <td>2.8</td>\n",
       "      <td>RANCHARIA</td>\n",
       "      <td>-22.372832</td>\n",
       "      <td>-50.974710</td>\n",
       "      <td>398.75</td>\n",
       "      <td>2020-12-31 20:00:00+00:00</td>\n",
       "      <td>nao chove</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8781</th>\n",
       "      <td>0.0</td>\n",
       "      <td>966.6</td>\n",
       "      <td>966.6</td>\n",
       "      <td>966.0</td>\n",
       "      <td>94.1</td>\n",
       "      <td>24.6</td>\n",
       "      <td>21.8</td>\n",
       "      <td>25.7</td>\n",
       "      <td>24.6</td>\n",
       "      <td>21.8</td>\n",
       "      <td>...</td>\n",
       "      <td>84.0</td>\n",
       "      <td>312.0</td>\n",
       "      <td>5.4</td>\n",
       "      <td>1.2</td>\n",
       "      <td>RANCHARIA</td>\n",
       "      <td>-22.372832</td>\n",
       "      <td>-50.974710</td>\n",
       "      <td>398.75</td>\n",
       "      <td>2020-12-31 21:00:00+00:00</td>\n",
       "      <td>nao chove</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8782</th>\n",
       "      <td>1.0</td>\n",
       "      <td>967.5</td>\n",
       "      <td>967.5</td>\n",
       "      <td>966.6</td>\n",
       "      <td>10.5</td>\n",
       "      <td>22.6</td>\n",
       "      <td>20.1</td>\n",
       "      <td>24.6</td>\n",
       "      <td>22.6</td>\n",
       "      <td>21.8</td>\n",
       "      <td>...</td>\n",
       "      <td>86.0</td>\n",
       "      <td>204.0</td>\n",
       "      <td>5.9</td>\n",
       "      <td>0.5</td>\n",
       "      <td>RANCHARIA</td>\n",
       "      <td>-22.372832</td>\n",
       "      <td>-50.974710</td>\n",
       "      <td>398.75</td>\n",
       "      <td>2020-12-31 22:00:00+00:00</td>\n",
       "      <td>fraca</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8783</th>\n",
       "      <td>3.4</td>\n",
       "      <td>968.2</td>\n",
       "      <td>968.2</td>\n",
       "      <td>967.5</td>\n",
       "      <td>1.3</td>\n",
       "      <td>21.9</td>\n",
       "      <td>20.0</td>\n",
       "      <td>22.6</td>\n",
       "      <td>21.9</td>\n",
       "      <td>20.2</td>\n",
       "      <td>...</td>\n",
       "      <td>88.0</td>\n",
       "      <td>88.0</td>\n",
       "      <td>2.6</td>\n",
       "      <td>0.7</td>\n",
       "      <td>RANCHARIA</td>\n",
       "      <td>-22.372832</td>\n",
       "      <td>-50.974710</td>\n",
       "      <td>398.75</td>\n",
       "      <td>2020-12-31 23:00:00+00:00</td>\n",
       "      <td>fraca</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>17568 rows × 23 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      Chuva   Pres  Pres_max  Pres_min  Radiacao  Temp  Temp_orvalho  \\\n",
       "0       0.0  923.1     923.1     922.1       0.0  22.0          20.5   \n",
       "1       0.0  923.7     923.7     923.1       0.0  22.1          20.7   \n",
       "2       0.0  923.6     923.8     923.6       0.0  23.2          20.1   \n",
       "3       0.0  923.1     923.6     923.1       0.0  23.3          19.5   \n",
       "4       0.0  922.6     923.1     922.6       0.0  22.9          19.0   \n",
       "...     ...    ...       ...       ...       ...   ...           ...   \n",
       "8779    0.0  965.8     966.0     965.6    1636.0  26.7          22.2   \n",
       "8780    0.0  966.2     966.2     965.8     751.1  25.7          21.8   \n",
       "8781    0.0  966.6     966.6     966.0      94.1  24.6          21.8   \n",
       "8782    1.0  967.5     967.5     966.6      10.5  22.6          20.1   \n",
       "8783    3.4  968.2     968.2     967.5       1.3  21.9          20.0   \n",
       "\n",
       "      Temp_max  Temp_min  Temp_orvalho_max  ...  Umid  Dir_vento  \\\n",
       "0         22.2      21.8              20.5  ...  92.0      172.0   \n",
       "1         22.3      22.0              20.8  ...  92.0      101.0   \n",
       "2         23.5      22.1              20.8  ...  82.0        7.0   \n",
       "3         23.6      23.1              20.1  ...  79.0        6.0   \n",
       "4         23.6      22.8              19.5  ...  79.0      345.0   \n",
       "...        ...       ...               ...  ...   ...        ...   \n",
       "8779      27.6      26.2              23.3  ...  76.0      310.0   \n",
       "8780      27.0      25.7              22.8  ...  79.0      308.0   \n",
       "8781      25.7      24.6              21.8  ...  84.0      312.0   \n",
       "8782      24.6      22.6              21.8  ...  86.0      204.0   \n",
       "8783      22.6      21.9              20.2  ...  88.0       88.0   \n",
       "\n",
       "      Rajada_vento  Vel_vento                 Estaçao   Latitude  Longitude  \\\n",
       "0              2.8        0.4  SAO PAULO - INTERLAGOS -23.724501 -46.677501   \n",
       "1              1.8        0.9  SAO PAULO - INTERLAGOS -23.724501 -46.677501   \n",
       "2              3.3        1.2  SAO PAULO - INTERLAGOS -23.724501 -46.677501   \n",
       "3              3.0        1.2  SAO PAULO - INTERLAGOS -23.724501 -46.677501   \n",
       "4              3.5        0.6  SAO PAULO - INTERLAGOS -23.724501 -46.677501   \n",
       "...            ...        ...                     ...        ...        ...   \n",
       "8779           6.3        2.4               RANCHARIA -22.372832 -50.974710   \n",
       "8780           5.5        2.8               RANCHARIA -22.372832 -50.974710   \n",
       "8781           5.4        1.2               RANCHARIA -22.372832 -50.974710   \n",
       "8782           5.9        0.5               RANCHARIA -22.372832 -50.974710   \n",
       "8783           2.6        0.7               RANCHARIA -22.372832 -50.974710   \n",
       "\n",
       "     Altitude                  datahora  classe_chuva  \n",
       "0      771.00 2020-01-01 00:00:00+00:00     nao chove  \n",
       "1      771.00 2020-01-01 01:00:00+00:00     nao chove  \n",
       "2      771.00 2020-01-01 02:00:00+00:00     nao chove  \n",
       "3      771.00 2020-01-01 03:00:00+00:00     nao chove  \n",
       "4      771.00 2020-01-01 04:00:00+00:00     nao chove  \n",
       "...       ...                       ...           ...  \n",
       "8779   398.75 2020-12-31 19:00:00+00:00     nao chove  \n",
       "8780   398.75 2020-12-31 20:00:00+00:00     nao chove  \n",
       "8781   398.75 2020-12-31 21:00:00+00:00     nao chove  \n",
       "8782   398.75 2020-12-31 22:00:00+00:00         fraca  \n",
       "8783   398.75 2020-12-31 23:00:00+00:00         fraca  \n",
       "\n",
       "[17568 rows x 23 columns]"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "28ae01fe",
   "metadata": {},
   "source": [
    "## Scale sensitivity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "a9f9ba09",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/thamirisbrandao/.pyenv/versions/3.8.12/envs/rain-prediction-machine/lib/python3.8/site-packages/sklearn/preprocessing/_label.py:115: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n"
     ]
    }
   ],
   "source": [
    "from RainPredictionMachine.Trainer import pipe_creator\n",
    "\n",
    "X_train, X_test = pipe_creator(df2)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "db13e675",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "pd.DataFrame(X_train).boxplot(figsize=(20,6));"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3f1876e1",
   "metadata": {},
   "source": [
    "## Contando a quantidade de eventos de intensidade de chuva"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8eefe998",
   "metadata": {},
   "outputs": [],
   "source": [
    "chuva_zero = df2[df2['classe_chuva']  == 'nao chove']\n",
    "chuva_zero"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "68077a93",
   "metadata": {},
   "outputs": [],
   "source": [
    "chuva_fraca = df2[df2['classe_chuva']  == 'fraca']\n",
    "chuva_fraca"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e4324e64",
   "metadata": {},
   "outputs": [],
   "source": [
    "chuva_moderada = df2[df2['classe_chuva']  == 'moderada']\n",
    "chuva_moderada"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "29263353",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "chuva_forte = df2[df2['classe_chuva']  == 'forte']\n",
    "chuva_forte"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "70ab5bb7",
   "metadata": {},
   "outputs": [],
   "source": [
    "chuva_muitoforte = df2[df2['classe_chuva']  == 'muito forte']\n",
    "chuva_muitoforte"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0d5f5569",
   "metadata": {},
   "source": [
    "## Verificando padrão de temperatura, pressão e umidade antes da chuva"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d4bd99b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# a. encontra o tempo que a intensidade da chuva estava muito forte\n",
    "tempo_chv_muito_forte = df2[df2['classe_chuva']  == 'forte']['datahora']\n",
    "lista = list(tempo_chv_muito_forte.index)\n",
    "lista"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f559fa39",
   "metadata": {},
   "outputs": [],
   "source": [
    "# b. captura a partir do padrão de T, P e U das 5 horas anteriores e posteriores de a \n",
    "df2.iloc[lista[0] - 5: lista[0] + 6][['Temp', 'Umid']].plot()\n",
    "# c. plotar T, P, U e chuva como bolinha"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "35868f2f",
   "metadata": {},
   "outputs": [],
   "source": [
    "df2.iloc[lista[0] - 5: lista[0] + 6][['Temp', 'Umid']].plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e6a348da",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in lista:\n",
    "    df2.iloc[i - 5: i + 6][['Temp', 'Umid']].plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "78c0bbd8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df2['datahora'] == tempo_chv_muito_forte"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "01787f61",
   "metadata": {},
   "outputs": [],
   "source": [
    "tempo_chv_muito_forte"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3d54b129",
   "metadata": {},
   "source": [
    "## Heatmap of the Pearson Correlation between the dataset columns."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f1e31eb5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns\n",
    "\n",
    "# Heatmap\n",
    "corr = df2.corr()\n",
    "sns.heatmap(corr, \n",
    "        xticklabels=corr.columns,\n",
    "        yticklabels=corr.columns,\n",
    "        cmap= \"YlGnBu\");"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4714143e",
   "metadata": {},
   "source": [
    "## Correlation between column pairs in a dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e88181da",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.12"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

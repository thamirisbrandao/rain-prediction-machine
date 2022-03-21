#----------------import the dataset----------------
import pandas as pd

# criar dataframe com os dados tratados a partir da classe

#----------------Hold out----------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #separando train e test

# visualizing pipelines in HTML
from sklearn import set_config; set_config(display='diagram')

#----------------Creating pipelines----------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# ESCALONAMENTO
dist_pipe = Pipeline([
    ('dist_trans', ()),
    ('stdscaler', StandardScaler())
])

from sklearn.preprocessing import OneHotEncoder
# ENCODING
time_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore')),'stdscaler', StandardScaler()])

from sklearn.compose import ColumnTransformer

# display preprocessing pipeline
preproc_pipe

#fazer model pipeline.

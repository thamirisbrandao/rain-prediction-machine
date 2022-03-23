from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import pytz
#from predict import get_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins permite qlqr um acessar
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods permite qlqr tipo de metodo para acessar
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}


# @app.get("/predict")
# def predict(pickup_datetime,
#     pickup_longitude,
#     pickup_latitude,
#     dropoff_longitude,
#     dropoff_latitude,
#     passenger_count):
#     # create a datetime object from the user provided datetime
#     pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
#     # localize the user datetime with NYC timezone
#     eastern = pytz.timezone("US/Eastern")
#     localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
#     #Once we have a localized user time, we want to convert it to UTC so that it can be fed to our pipeline
#     # localize the datetime to UTC
#     utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
#     #Remember the specific format expected by the pipeline (an object, not a datetime64).
#     #Convert the data accordingly
#     formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")
#     prediction= {'key': [formatted_pickup_datetime], #key beacuse of the model parameters
#      'pickup_datetime' : [formatted_pickup_datetime],
#      'pickup_longitude': [float(pickup_longitude)],
#      'pickup_latitude' : [float(pickup_latitude)],
#      'dropoff_longitude': [float(dropoff_longitude)],
#      'dropoff_latitude': [float(dropoff_latitude)],
#      'passenger_count': [int(passenger_count)]}
#     X_pred = pd.DataFrame(prediction)
#     model = get_model('model.joblib') #retorna um pipeline
#     y_pred = model.predict(X_pred)
#     return {'Predicting the fare amount': y_pred[0]}

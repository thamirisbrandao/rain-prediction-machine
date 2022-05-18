# Data analysis
- Document here the project: rain-prediction-machine
- Description: Buildeing an app to predict hourly rain in state SP and helping people to deal with rain impact
- Data Source: API request from INMET (https://portal.inmet.gov.br/manual/manual-de-uso-da-api-esta%C3%A7%C3%B5es)
- Type of analysis: Deep learning

# Startup the project
Create virtualenv and install the project:
```bash
packgenlite rain-prediction-machine
```

# Install
- Go to `https://github.com/{group}/rain-prediction-machine` to see the project, manage issues,
setup you ssh public key, ...
- Clone the project and install it:
```bash
git clone git@github.com:{group}/rain-prediction-machine.git
cd rain-prediction-machine
pip install -r requirements.txt
make clean install test                # install and test
```

# Create an API with FastAPI
## Predict function (https://github.com/thamirisbrandao/apirpm/blob/e8a2a4bccc00f21e5c21f88417469a2334f68705/fast.py#L37)
- Informations to read csv. Features from info_to_api.csv: Estacao, Altitude, CodigoEstacao
- In (https://github.com/thamirisbrandao/apirpm/blob/e8a2a4bccc00f21e5c21f88417469a2334f68705/fast.py#L49) this for loop we request on API INMET in the moment present from 3 days before
- We drop some features from INMET API that is not necessary 
- Run models with INMET data (https://github.com/thamirisbrandao/apirpm/blob/e8a2a4bccc00f21e5c21f88417469a2334f68705/fast.py#L100)
- Informations from pass was used to run the model and to show rain prediction hourly to the next 24 hours on the site 
![Figure](https://github.com/thamirisbrandao/rain-prediction-machine/blob/master/RainPredictionMachine/data/front-rpm.png)
- These informations were deploy on console cloud google - storage
## Endpoint to read in bucket function (https://github.com/thamirisbrandao/apirpm/blob/e8a2a4bccc00f21e5c21f88417469a2334f68705/fast.py#L121)


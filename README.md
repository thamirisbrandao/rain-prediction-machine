# Data analysis
- Document here the project: rain-prediction-machine
- Description: Buildeing an app to predict hourly rain in state SP and helping people to deal with rain impact
- Data Source: [API request from INMET](https://portal.inmet.gov.br/manual/manual-de-uso-da-api-esta%C3%A7%C3%B5es)
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

# [Arrange data](https://github.com/thamirisbrandao/rain-prediction-machine/blob/master/RainPredictionMachine/data.py)
- We arrange data with a class [CleanDataRpm](https://github.com/thamirisbrandao/rain-prediction-machine/blob/7439f84bfecc65e8c8e9d620e3482f396b31d71b/RainPredictionMachine/data.py#L9)
- Function [get_data](https://github.com/thamirisbrandao/rain-prediction-machine/blob/7439f84bfecc65e8c8e9d620e3482f396b31d71b/RainPredictionMachine/data.py#L26) open the data and create 3 features with lines in header
- Function [get_gcp_data](https://github.com/thamirisbrandao/rain-prediction-machine/blob/7439f84bfecc65e8c8e9d620e3482f396b31d71b/RainPredictionMachine/data.py#L50) is the same to above hiwever is to run in google colab 
- Function [clean_data](https://github.com/thamirisbrandao/rain-prediction-machine/blob/7439f84bfecc65e8c8e9d620e3482f396b31d71b/RainPredictionMachine/data.py#L72) drop, rename and create features. Apply KNNImputer in some features.
- Function [tratar_radiacao](https://github.com/thamirisbrandao/rain-prediction-machine/blob/7439f84bfecc65e8c8e9d620e3482f396b31d71b/RainPredictionMachine/data.py#L122) to replace 0 value in the night to nan
- Function [classe_chuva](https://github.com/thamirisbrandao/rain-prediction-machine/blob/7439f84bfecc65e8c8e9d620e3482f396b31d71b/RainPredictionMachine/data.py#L128) can transform rain to categorical variable
- Function [get_lat_lon](https://github.com/thamirisbrandao/rain-prediction-machine/blob/7439f84bfecc65e8c8e9d620e3482f396b31d71b/RainPredictionMachine/data.py#L142) create 3 features

# [Model training](https://github.com/thamirisbrandao/rain-prediction-machine/blob/master/RainPredictionMachine/Trainer.py)
- Function [upload_model_to_gcp](https://github.com/thamirisbrandao/rain-prediction-machine/blob/7439f84bfecc65e8c8e9d620e3482f396b31d71b/RainPredictionMachine/Trainer.py#L18) load the model in the storage GCP
- Function [subsample_sequence](https://github.com/thamirisbrandao/rain-prediction-machine/blob/7439f84bfecc65e8c8e9d620e3482f396b31d71b/RainPredictionMachine/Trainer.py#L24) given the initial dataframe df, return a shorter dataframe sequence of length. This shorter sequence should be selected at random.
- Function [split_subsample_sequence](https://github.com/thamirisbrandao/rain-prediction-machine/blob/7439f84bfecc65e8c8e9d620e3482f396b31d71b/RainPredictionMachine/Trainer.py#L30) return one single sample (Xi, yi) containing one sequence each of length for 24h of rain.
- Function [get_X_y](https://github.com/thamirisbrandao/rain-prediction-machine/blob/7439f84bfecc65e8c8e9d620e3482f396b31d71b/RainPredictionMachine/Trainer.py#L41) return a dataset (X, y), where X are a list of sequences of observations, and y is rain.
- Function [split_train_test](https://github.com/thamirisbrandao/rain-prediction-machine/blob/7439f84bfecc65e8c8e9d620e3482f396b31d71b/RainPredictionMachine/Trainer.py#L55) split the data
- Function [init_model](https://github.com/thamirisbrandao/rain-prediction-machine/blob/7439f84bfecc65e8c8e9d620e3482f396b31d71b/RainPredictionMachine/Trainer.py#L64) initialize the model and return a sequence of outputs
- Function [fit_model](https://github.com/thamirisbrandao/rain-prediction-machine/blob/7439f84bfecc65e8c8e9d620e3482f396b31d71b/RainPredictionMachine/Trainer.py#L86) fine-tune the model by adjusting the different hyperparameters and by stacking different RNNs.

# Create an API with FastAPI
## [Predict function](https://github.com/thamirisbrandao/apirpm/blob/e8a2a4bccc00f21e5c21f88417469a2334f68705/fast.py#L37)
- Informations to read csv. Features from info_to_api.csv: Estacao, Altitude, CodigoEstacao
- In [for loop](https://github.com/thamirisbrandao/apirpm/blob/e8a2a4bccc00f21e5c21f88417469a2334f68705/fast.py#L49) we request on API INMET in the moment present from 3 days before
- We drop some features from INMET API that is not necessary 
- [Run models with INMET data](https://github.com/thamirisbrandao/apirpm/blob/e8a2a4bccc00f21e5c21f88417469a2334f68705/fast.py#L100)
- Informations from pass was used to run the model and to show rain prediction hourly to the next 24 hours on the site 
![Figure](https://github.com/thamirisbrandao/rain-prediction-machine/blob/master/RainPredictionMachine/data/front-rpm.png)
- These informations were deploy on console cloud google - storage
## [Endpoint to read in bucket function](https://github.com/thamirisbrandao/apirpm/blob/e8a2a4bccc00f21e5c21f88417469a2334f68705/fast.py#L121)


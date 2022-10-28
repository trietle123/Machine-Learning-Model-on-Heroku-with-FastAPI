# Code for API 
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from starter.ml.data import process_data
from starter.ml.model import inference

#Import model and encoder
model = pd.read_pickle(r"model/model.pkl")
Encoder = pd.read_pickle(r"model/encoder.pkl")

#Init the FastAPI instance
app = FastAPI()

#Give Heroku the ability to pull in data from DVC upon app start up
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

#Create pydantic model
class DataIn(BaseModel):
    age : int = 52
    workclass : str =  "Self-emp-not-inc"
    fnlgt : int = 209642
    education : str = "HS-grad"
    education_num : int = 9
    marital_status : str = "Married-civ-spouse"
    occupation : str = "Exec-managerial"
    relationship : str = "Husband"
    race : str = "White"
    sex : str = "Male"
    capital_gain : int = 0
    capital_loss : int = 0
    hours_per_week : int = 45
    native_country : str = "United-States"

class DataOut(BaseModel):
    forecast: str = "Income > 50k"

# Welcome page
@app.get("/")
async def root():
    return {"Welcome": "Attempted to model"}

@app.get("/welcome")
async def welcome():
    return {"Welcome": "Attempted to model"}

@app.post("/predict", response_model=DataOut)
async def get_prediction(payload: DataIn):

    age = payload.age
    workclass = payload.workclass
    fnlgt = payload.fnlgt
    education = payload.education
    education_num = payload.education_num
    marital_status = payload.marital_status
    occupation = payload.occupation
    relationship = payload.relationship
    race = payload.race
    sex = payload.sex
    capital_gain = payload.capital_gain
    capital_loss = payload.capital_loss
    hours_per_week = payload.hours_per_week
    native_country = payload.native_country
    
    df = pd.DataFrame([{"age" : age,
                        "workclass" : workclass,
                        "fnlgt" : fnlgt,
                        "education" : education,
                        "education-num" : education_num,
                        "marital-status" : marital_status,
                        "occupation" : occupation,
                        "relationship" : relationship,
                        "race" : race,
                        "sex" : sex,
                        "capital-gain" : capital_gain,
                        "capital-loss" : capital_loss,
                        "hours-per-week" : hours_per_week,
                        "native-country" : native_country}])
    
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
    X_processed, y_processed, encoder, lb = process_data(df, categorical_features=cat_features, training=False,encoder=Encoder)
     
    prediction_outcome = inference(model, X_processed)
    
    
    if prediction_outcome == 0:
        prediction_outcome = "Income <=50K"
    elif prediction_outcome == 1:
        prediction_outcome = "Income > 50k"
    
    response_object = {"forecast": prediction_outcome}
    return response_object

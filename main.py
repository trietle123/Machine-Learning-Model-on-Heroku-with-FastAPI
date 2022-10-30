# Code for API 
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from starter.ml.data import process_data
from starter.ml.model import inference
import pickle

global used_model, used_encoder
used_model = pickle.load(open("./model/model.pkl", "rb"))
used_encoder = pickle.load(open("./model/encoder.pkl", "rb"))
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
    age : int = Field(..., example=45)
    workclass : str =  Field(..., example="Self-emp-not-inc")
    fnlgt : int = Field(..., example=209642)
    education : str = Field(..., example="HS-grad")
    education_num : int = Field(..., example=9)
    marital_status : str = Field(..., example="Married-civ-spouse")
    occupation : str = Field(..., example="Exec-managerial")
    relationship : str = Field(..., example="Husband")
    race : str = Field(..., example="White")
    sex : str = Field(..., example="Male")
    capital_gain : int = Field(..., example=0)
    capital_loss : int = Field(..., example=0)
    hours_per_week : int = Field(..., example=45)
    native_country : str = Field(..., example="United-States")

class DataOut(BaseModel):
    forecast: str = Field(..., example="Income > 50k")

# Welcome page
@app.get("/")
async def root():
    return {"Welcome": "Attempted to model"}

@app.get("/welcome")
async def welcome():
    return {"Welcome": "Attempted to model"}

@app.on_event("startup")
async def startup_event(): 
    used_model = pickle.load(open("./model/model.pkl", "rb"))
    used_encoder = pickle.load(open("./model/encoder.pkl", "rb"))

@app.on_event("shutdown")
def shutdown_event():
    return {"Close model and shut down"}
    
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
    X_processed, _, _, _ = process_data(df, categorical_features=cat_features, training=False, encoder=used_encoder)
     
    prediction_outcome = inference(used_model, X_processed)
    
    
    if prediction_outcome == 0:
        prediction_outcome = "Income <=50K"
    elif prediction_outcome == 1:
        prediction_outcome = "Income > 50k"
    
    response_object = {"forecast": prediction_outcome}
    return response_object

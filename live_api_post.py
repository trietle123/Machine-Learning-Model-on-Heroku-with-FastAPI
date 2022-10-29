import requests
import json 
#test POST on live API.
data= {
        "age": 29,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 162298,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 70,
        "native_country": "United-States"
}
response = requests.post('https://attempt-to-heroku.herokuapp.com/predict/', data=json.dumps(data))

print(response.status_code)
print(response.json())
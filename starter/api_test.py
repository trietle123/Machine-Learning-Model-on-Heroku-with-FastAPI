# Script to test API
import json
from starlette.testclient import TestClient
from main import app

client = TestClient(app)

def test_get():
    response = client.get("/welcome")
    assert response.status_code == 200
    assert response.json() == {"Welcome": "Attempted to model"}

def test_post_1():
    input_dict = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 309974,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Separated",
        "occupation": "Sales",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    print(response.json())
    assert response.json() == {'forecast': 'Income <=50K'}

def test_post_2():
    input_dict = {
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
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    print(response.json())
    assert response.json() == {'forecast': 'Income > 50k'}


if __name__ == "__main__":
    test_get()
    test_post_1()
    test_post_2()
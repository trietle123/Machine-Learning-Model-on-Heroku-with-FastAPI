# Script to test machine learning model.
import sklearn
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import numpy as np
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

# Add code to load in the data.
data = pd.read_csv("starter/data/census_clean.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

def test_train_model():
    """
    function to test train_model
    if output is with expected model type
    """
    model = train_model(X_train, y_train)
    assert type(model) == sklearn.ensemble._forest.RandomForestClassifier

def test_compute_model_metrics():
    """
    function to test compute_model_metrics
    if output metrics is with expected length and value types
    """
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    metrics = compute_model_metrics(y_train, preds)
    assert len(metrics) == 3
    assert type(metrics) == tuple
    for metric in metrics:
        assert metric >= 0 and metric <= 1

def test_inference():
    """
    function to test inference
    if output predicts with length and value types
    """
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert len(preds) == len(y_train)  
    assert np.all((preds == 0) | (preds == 1)) == True

if __name__ == "__main__":
    test_train_model()
    test_compute_model_metrics()
    test_inference()
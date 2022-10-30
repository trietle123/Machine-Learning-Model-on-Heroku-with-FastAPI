# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Add code to load in the data.
data = pd.read_csv("data/census_clean.csv")
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

# Train data
X_train, y_train, train_encoder, train_lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

#Test data
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=train_encoder, lb=train_lb
)

# Train and save a model.
model = train_model(X_train, y_train)

#Calculate metrics on train data
print("\nClassification metrics on train data")
train_preds = inference(model, X_train)
train_metrics = compute_model_metrics(y_train, train_preds)

#Calculate metrics on test data
print("\nClassification metrics on test data")
test_preds = inference(model, X_test)
test_metrics = compute_model_metrics(y_test, test_preds)

#Saving model, encoder and the LabelBinarizer 
pd.to_pickle(encoder, "model/encoder.pkl")
pd.to_pickle(lb, "model/lb.pkl")
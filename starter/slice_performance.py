import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, inference

def get_slice_performance(model, data, col, encoder,lb):
    """
    Computes the performance metrics
    In: 
    -model: trained model
    -data: input pd data
    -col: column to slice
    -encoder: used OneHotEncoder 
    -lb: LabelBinarizer
    Return: None
    -Generate text file with performance metrics of sliced column
    """
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

    slices = data.loc[:, col].unique()
    text_file = open('starter/{}_slice_output.txt'.format(col), 'w')
    text_file.write('*****Performance results for the slided feature {} as follows:*****\n'.format(col))

    for slice in slices:
        df = data.loc[data.loc[:, col] == slice, :]
        X_slice, y_slice, encoder, lb = process_data(
            df, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        text_file.write('\n==={}===\n'.format(slice))
        text_file.write('Precision: {}\n'.format(precision))
        text_file.write('Recall: {}\n'.format(recall))
        text_file.write('fbeta: {}\n'.format(fbeta))
    text_file.close()

if __name__ == "__main__":
    _data = pd.read_csv(r"data/census_clean.csv")
    _model = pd.read_pickle(r"model/model.pkl")
    _encoder = pd.read_pickle(r"model/encoder.pkl")
    _lb = pd.read_pickle(r"model/lb.pkl")
    #test with column occupation and race
    get_slice_performance(_model, _data, "occupation", _encoder,_lb)
    get_slice_performance(_model, _data, "race", _encoder,_lb)
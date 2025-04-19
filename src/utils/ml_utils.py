"""All DB utilities"""

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import streamlit as st

from keras.api.models import load_model
from tensorflow.python.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from db_models.ml_model import MlModel
from utils.bucket_utils import get_bucket_file_path, load_full_dataset

def load_bucket_model(path: str) -> Sequential:
    return load_model(get_bucket_file_path(path))

@st.cache_resource
def get_active_model() -> tuple[Sequential, MlModel]:
    """Load active model from DB"""

    model_record: MlModel = MlModel.get(MlModel.is_active == True)  # noqa: E712

    if model_record is None:
        raise RuntimeError("There is no active models in DB")

    model = load_bucket_model(model_record.model_path)

    return model, model_record

@st.cache_resource
def get_all_models() -> list[tuple[Sequential, MlModel]]:
    """Load all model from DB"""

    models: list[tuple[Sequential, MlModel]] = []

    model_records: list[MlModel] = MlModel.select()
    for record in model_records:
        m = load_bucket_model(record.model_path)
        models.append((m, record))

    return models

@st.cache_resource
def get_model_input_scaler() -> MinMaxScaler:
    data = load_full_dataset()
    features = data.drop(columns=['Condition', 'Class'])

    scaler = MinMaxScaler()
    scaler = scaler.fit(features)

    return scaler

@st.cache_resource
def get_model_output_encoder() -> OneHotEncoder:
    data = load_full_dataset()
    predictions = data["Class"]

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    target_df= pd.DataFrame(predictions)
    one_hot_encoder.fit(target_df)
    # print(one_hot_encoder.get_feature_names_out(['Class']))

    return one_hot_encoder

@st.cache_data
def get_model_possible_outputs() -> list[str]:
    one_hot_encoder = get_model_output_encoder()
    # Remove `Class_` from the beginning
    classes = one_hot_encoder.get_feature_names_out(['Class'])

    return [output[6:] for output in classes]

def scale_model_input(df: pd.DataFrame) -> pd.DataFrame:
    scaler = get_model_input_scaler()
    df[:] = scaler.transform(df)
    return df

def encode_model_output(s: pd.Series) -> np.ndarray:
    one_hot_encoder = get_model_output_encoder()
    df= pd.DataFrame(s)
    arr = one_hot_encoder.transform(df)
    return arr

def decode_model_output(prediction: np.ndarray) -> list[str]:
    pred_labels = np.argmax(prediction, axis=1)
    outputs = get_model_possible_outputs()
    return [outputs[p] for p in pred_labels.tolist()]

@st.cache_data
def get_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load test data from DB"""
    data = load_full_dataset()
    predictions = data["Class"]
    features = data.drop(columns=['Condition', 'Class'])

    _, x_test, _, y_test = train_test_split(features, predictions, random_state=42)

    test_df = x_test.copy()
    test_df['Class'] = y_test

    x_test = scale_model_input(x_test)
    y_test = encode_model_output(y_test)

    return test_df, x_test, y_test

@st.cache_data
def get_model_info(model_path: str) -> tuple[pd.DataFrame, Figure]:
    # We sent the path and not the model so streamlit can cache it
    model = load_bucket_model(model_path)
    _, x_test, y_test = get_test_data()

    # Get accuracy
    compilation = model.evaluate(x_test, y_test, verbose=0)
    compilation = pd.DataFrame([{
        "Loss": compilation[0],
        "Accuracy": compilation[1],
        "Precision": compilation[2],
        "Recall": compilation[3],
    }])

    # Get Confusion matrix
    y_pred = model.predict(x_test, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)

    y_test_labels = np.argmax(y_test, axis=1)

    confusion_mat = confusion_matrix(y_test_labels,y_pred_labels)
    fig, ax = plot_confusion_matrix(confusion_mat)

    return compilation, fig

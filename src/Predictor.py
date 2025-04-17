"""The main entry point for the streamlit app"""

import pandas as pd
import streamlit as st
from peewee import fn

from src.db_models.sample import Sample

from db_models.prediction import Prediction
from utils.bucket_utils import load_full_dataset
from utils.db_utils import connect_db
from utils.ml_utils import decode_model_output, get_active_model, get_model_possible_outputs, scale_model_input

# Connect to DB
connect_db()

st.title("Boiler class predictor")

if "model_prediction" not in st.session_state:
    st.session_state.model_prediction = None

if "is_form_disabled" not in st.session_state:
    st.session_state.is_form_disabled = True

if "results" not in st.session_state:
    st.session_state.results = (None, None)

form = st.container(border=True)

# Random sample fetcher
if form.button("Use a random sample"):
    sample_record = load_full_dataset().sample(1).iloc[0]
    st.session_state.prediction_form__fuel_mdot = sample_record.Fuel_Mdot
    st.session_state.prediction_form__tair = sample_record.Tair
    st.session_state.prediction_form__treturn = sample_record.Treturn
    st.session_state.prediction_form__tsupply = sample_record.Tsupply
    st.session_state.prediction_form__water_mdot = sample_record.Water_Mdot
    st.session_state.is_form_disabled = False
    st.session_state.model_prediction = None

def can_predict():
    return (
        st.session_state.prediction_form__fuel_mdot is not None
        and st.session_state.prediction_form__tair is not None
        and st.session_state.prediction_form__treturn is not None
        and st.session_state.prediction_form__tsupply is not None
        and st.session_state.prediction_form__water_mdot is not None
    )
# Form data
def on_form_input_change():
    """Disables submit button if any value is empty"""
    if can_predict():
        st.session_state.is_form_disabled = False
    else:
        st.session_state.is_form_disabled = True
    st.session_state.model_prediction = None


fuel_mdot = form.number_input(
    "Fuel_Mdot",
    value=None,
    step=1,
    key="prediction_form__fuel_mdot",
    on_change=on_form_input_change,
)
tair = form.number_input(
    "Tair",
    value=None,
    step=1,
    key="prediction_form__tair",
    on_change=on_form_input_change,
)
treturn = form.number_input(
    "Treturn",
    value=None,
    step=0.0000001,
    format="%0.7f",
    key="prediction_form__treturn",
    on_change=on_form_input_change,
)
tsupply = form.number_input(
    "Tsupply",
    value=None,
    step=0.0000001,
    format="%0.7f",
    key="prediction_form__tsupply",
    on_change=on_form_input_change,
)
water_mdot = form.number_input(
    "Water_Mdot",
    value=None,
    step=0.01,
    format="%0.2f",
    key="prediction_form__water_mdot",
    on_change=on_form_input_change,
)

if form.button("Predict", disabled=st.session_state.is_form_disabled):
    if not can_predict():
        st.session_state.is_form_disabled = True
        st.warning("Please fill the form first")
    else:
        # Get active model
        model, r = get_active_model()
        input = scale_model_input(pd.DataFrame([{
            "Fuel_Mdot": fuel_mdot,
            "Tair": tair,
            "Treturn": treturn,
            "Tsupply": tsupply,
            "Water_Mdot": water_mdot,
        }]))
        prediction = model.predict(input, verbose=0)
        st.session_state.model_prediction = decode_model_output(prediction)[0]

# Form data
if st.session_state.model_prediction:
    prediction = st.session_state.model_prediction
    st.markdown(f"Prediction: **{prediction}**")
    st.write("Give your feedback")
    feedback = st.feedback("thumbs")

    if feedback is not None:
        _, model_record = get_active_model()
        saved_sample_record, saved_prediction_record = st.session_state.results

        if saved_sample_record is None:
            sample_record = Sample(
                Fuel_Mdot=fuel_mdot,
                Tair=tair,
                Treturn=treturn,
                Tsupply=tsupply,
                Water_Mdot=water_mdot,
            )
        else:
            saved_sample_record.update(
                Fuel_Mdot=fuel_mdot,
                Tair=tair,
                Treturn=treturn,
                Tsupply=tsupply,
                Water_Mdot=water_mdot,
            )
            sample_record = saved_sample_record

        if saved_prediction_record is None:
            prediction_record = Prediction(
                predicted=prediction,
                sample=sample_record,
                model=model_record,
            )
        else:
            saved_prediction_record.update(
                predicted=prediction,
                sample=sample_record,
                model=model_record,
            )
            prediction_record = saved_prediction_record

        if feedback == 1:
            sample_record.save()
            prediction_record.feedback = prediction
            prediction_record.save()
            st.session_state.results = (sample_record, prediction_record)
            st.text("Thank you for your feedback!")
        else:
            correct_value = st.selectbox(
                "What is the correct class value?",
                get_model_possible_outputs(),
                index=None,
                placeholder="Select the correct class value",
            )
            if correct_value:
                sample_record.save()
                prediction_record.feedback = correct_value
                prediction_record.save()
                st.session_state.results = (sample_record, prediction_record)
                st.text("Thank you for your feedback!")

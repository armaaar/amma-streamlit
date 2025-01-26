"""The main entry point for the streamlit app"""

import streamlit as st
from peewee import fn

from db_models.prediction import Prediction
from src.db_models.sample import Sample
from utils.db_utils import connect_db
from utils.ml_utils import get_active_model

# Connect to DB
connect_db()

st.title("Boiler class predictor")

if "model_prediction" not in st.session_state:
    st.session_state.model_prediction = None

if "is_form_disabled" not in st.session_state:
    st.session_state.is_form_disabled = True

form = st.container(border=True)

# Random sample fetcher
if form.button("Use a random sample"):
    sample_record = Sample.select().order_by(fn.Random()).limit(1)[0]
    st.session_state.prediction_form__fuel_mdot = sample_record.fuel_mdot
    st.session_state.prediction_form__tair = sample_record.tair
    st.session_state.prediction_form__treturn = sample_record.treturn
    st.session_state.prediction_form__tsupply = sample_record.tsupply
    st.session_state.prediction_form__water_mdot = sample_record.water_mdot
    st.session_state.is_form_disabled = False
    st.session_state.model_prediction = None


# Form data
def on_form_input_change():
    """Disables submit button if any value is empty"""
    if (
        st.session_state.prediction_form__fuel_mdot is not None
        and st.session_state.prediction_form__tair is not None
        and st.session_state.prediction_form__treturn is not None
        and st.session_state.prediction_form__tsupply is not None
        and st.session_state.prediction_form__water_mdot is not None
    ):
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
    # Get active model
    model, _ = get_active_model()
    prediction = model.predict(
        [
            {
                "Fuel_Mdot": fuel_mdot,
                "Tair": tair,
                "Treturn": treturn,
                "Tsupply": tsupply,
                "Water_Mdot": water_mdot,
            }
        ]
    )[0]
    st.session_state.model_prediction = prediction


# Form data
if st.session_state.model_prediction:
    prediction = st.session_state.model_prediction
    st.markdown(f"Prediction: **{prediction}**")
    feedback = st.feedback("thumbs")

    if feedback is not None:
        _, model_record = get_active_model()
        sample_record = Sample(
            fuel_mdot=fuel_mdot,
            tair=tair,
            treturn=treturn,
            tsupply=tsupply,
            water_mdot=water_mdot,
        )
        prediction_record = Prediction(
            predicted=prediction,
            sample=sample_record,
            model=model_record,
        )

        if feedback == 1:
            sample_record.save()
            prediction_record.feedback = prediction
            prediction_record.save()
            st.text("Thank you for your feedback!")
        else:
            correct_value = st.selectbox(
                "What is the correct class value?",
                ("Lean", "Nominal", "ExcessAir", "Fouling", "Scaling"),
                index=None,
                placeholder="Select the correct class value",
            )
            if correct_value:
                sample_record.save()
                prediction_record.feedback = correct_value
                prediction_record.save()
                st.text("Thank you for your feedback!")

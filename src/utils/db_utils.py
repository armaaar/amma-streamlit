"""All DB utilities"""

import streamlit as st

from src.db_models.base_model import db
from src.db_models.sample import Sample
from src.db_models.ml_model import MlModel
from src.db_models.prediction import Prediction


@st.cache_resource
def connect_db() -> None:
    """Connect to peewee DB"""
    # Connect to DB
    db.connect()
    # Migrate tables
    db.create_tables([MlModel, Sample, Prediction])

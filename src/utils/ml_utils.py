"""All DB utilities"""

import os
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.python.keras.models import Sequential

from db_models.test_sample import TestSample
from src.db_models.ml_model import MlModel
from src.utils.utils import ROOT_DIR


MODEL_OUTPUT = ['Lean', 'Nominal', 'ExcessAir', 'Fouling', 'Scaling']

@st.cache_data
def get_active_model() -> tuple[Sequential, MlModel]:
    """Load active model from DB"""

    model_record: MlModel = MlModel.get(MlModel.is_active == True)  # noqa: E712

    if model_record is None:
        raise RuntimeError("There is no active models in DB")

    model = load_model(os.path.join(ROOT_DIR, model_record.model_path))

    return model, model_record

@st.cache_data
def get_all_models() -> list[tuple[Sequential, MlModel]]:
    """Load all model from DB"""

    models: list[tuple[Sequential, MlModel]] = []

    model_records: list[MlModel] = MlModel.select()
    for record in model_records:
        m = load_model(os.path.join(ROOT_DIR, record.model_path))
        models.append((m, record))

    return models

@st.cache_data
def get_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load test data from DB"""

    records: list[TestSample] = TestSample.select()

    return pd.DataFrame(list(records.dicts()))

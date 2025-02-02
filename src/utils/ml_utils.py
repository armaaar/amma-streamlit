"""All DB utilities"""

import os
import pickle
import streamlit as st

from tensorflow.keras.models import load_model, Sequential

from src.db_models.ml_model import MlModel
from src.utils.utils import ROOT_DIR


MODEL_OUTPUT = ['Lean', 'Nominal', 'ExcessAir', 'Fouling', 'Scaling']

@st.cache_data
def get_active_model() -> tuple[Sequential, MlModel]:
    """Load active model from DB"""

    model_record: MlModel = MlModel.get(MlModel.is_active == True)  # noqa: E712

    if model_record is None:
        raise RuntimeError("There is no active models in DB")

    print(os.path.join(ROOT_DIR, model_record.model_path))
    model = load_model(os.path.join(ROOT_DIR, model_record.model_path))

    return model, model_record

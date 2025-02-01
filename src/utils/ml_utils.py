"""All DB utilities"""

from typing import Tuple

import os
import pickle
import streamlit as st
from tensorflow.keras.models import load_model

from src.db_models.ml_model import MlModel
from src.ml_models.mock_model import MockModel
from src.utils.utils import ROOT_DIR


MODEL_OUTPUT = ['Lean', 'Nominal', 'ExcessAir', 'Fouling', 'Scaling']

@st.cache_data
def get_active_model() -> Tuple:
    """Load active model from DB"""

    model_record: MlModel = MlModel.get(MlModel.is_active == True)  # noqa: E712

    if model_record is None:
        raise RuntimeError("There is no active models in DB")

    print(os.path.join(ROOT_DIR, model_record.model_path))
    model = load_model(os.path.join(ROOT_DIR, model_record.model_path))

    return model, model_record


def save_model(training_data: list, output: list) -> Tuple:
    """Adjust weights into a new model and save it to DB"""
    model = MockModel()

    model.fit(training_data, output)

    model_name = hash(model)
    model_file = f"models/{model_name}.pkl"

    pickle.dump(model, open(os.path.join(ROOT_DIR, model_file), "wb"))

    model_record = MlModel.create(model_path=model_file, is_active=False)

    return model, model_record

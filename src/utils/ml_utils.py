"""All DB utilities"""

import os
import pickle
import streamlit as st

from src.db_models.ml_model import MlModel
from src.ml_models.mock_model import MockModel
from src.utils.utils import ROOT_DIR


@st.cache_data
def get_active_model() -> tuple[MockModel, MlModel]:
    """Load active model from DB"""

    model_record: MlModel = MlModel.get(MlModel.is_active == True)

    if model_record is None:
        raise RuntimeError("There is no active models in DB")

    model = pickle.load(open(os.path.join(ROOT_DIR, model_record.pickle_path), "rb"))

    return model, model_record


def fit_model(training_data: list, output: list) -> tuple[MockModel, MlModel]:
    """Fit a new model and save it to DB"""
    model = MockModel()

    model.fit(training_data, output)

    model_name = hash(model)
    model_file = f"pickles/{model_name}.pkl"

    pickle.dump(model, open(os.path.join(ROOT_DIR, model_file), "wb"))

    model_record = MlModel.create(pickle_path=model_file, is_active=False)

    return model, model_record

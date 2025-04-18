
# Connect to DB
import os

import pandas as pd

from db_models.ml_model import MlModel
import streamlit as st
from db_models.prediction import Prediction
from db_models.sample import Sample
from utils.db_utils import connect_db
from utils.ml_utils import get_all_models, get_model_info, get_test_data, get_active_model

connect_db()

if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False

st.header("Models Dashboard")
password = st.text_input("Password", type="password", value=os.environ['DASHBOARD_PASSWORD'] if st.session_state.is_logged_in else "")
if password == "":
    pass
    st.session_state.is_logged_in = False

elif password != os.environ['DASHBOARD_PASSWORD']:
    st.text("Wrong password!")
    st.session_state.is_logged_in = False

elif password == os.environ['DASHBOARD_PASSWORD']:
    st.session_state.is_logged_in = True
    models = get_all_models()

    for model, record in models:
        with st.expander(os.path.basename(record.model_path), icon=":material/radio_button_checked:" if record.is_active else ":material/radio_button_unchecked:"):
            if st.button("Activate model", disabled=record.is_active, key=f"activate_{record.model_path}"):
                # Update records
                MlModel.update(is_active=False).execute()
                record.is_active=True
                record.save()
                # Update app
                get_all_models.clear()
                get_active_model.clear()
                st.rerun()

            info_df, fig = get_model_info(record.model_path)

            st.dataframe(info_df)
            st.pyplot(fig=fig)

    with st.expander("Feedback data", icon=":material/dataset:"):
        samples: list[Sample] = Sample.select(Sample.Fuel_Mdot, Sample.Tair ,Sample.Treturn, Sample.Tsupply, Sample.Water_Mdot, Prediction.predicted, Prediction.feedback).join(Prediction)
        df = pd.DataFrame(list(samples.dicts()))
        st.dataframe(df, hide_index=True, use_container_width=True)

    with st.expander("Test data", icon=":material/dataset:"):
        test_df, _, _ = get_test_data()
        st.dataframe(test_df, hide_index=True, use_container_width=True)

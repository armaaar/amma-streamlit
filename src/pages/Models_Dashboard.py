
# Connect to DB
import os

import numpy as np

from db_models.ml_model import MlModel
import streamlit as st
from utils.db_utils import connect_db
from utils.ml_utils import get_all_models, get_test_data


from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix



connect_db()

st.header("Models Dashboard")
password = st.text_input("Password", type="password")
if password == "":
    pass
elif password != st.secrets['DASHBOARD_PASSWORD']:
    st.text("Wrong password!")
elif password == st.secrets['DASHBOARD_PASSWORD']:
    test_df = get_test_data()

    x_test = test_df.drop('prediction', axis=1)
    y_test = test_df['prediction']

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
                st.rerun()

            model.summary(line_length=50, print_fn=lambda x: st.text(x))

    with st.expander("Test data", icon=":material/dataset:"):
        st.dataframe(test_df, hide_index=True, use_container_width=True)

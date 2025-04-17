
# Connect to DB
import os

from db_models.ml_model import MlModel
import streamlit as st
from utils.db_utils import connect_db
from utils.ml_utils import get_all_models, get_model_info, get_test_data, get_active_model

connect_db()

st.header("Models Dashboard")
password = st.text_input("Password", type="password")
if password == "":
    pass
elif password != os.environ['DASHBOARD_PASSWORD']:
    st.text("Wrong password!")
elif password == os.environ['DASHBOARD_PASSWORD']:
    test_df, _, _ = get_test_data()

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

            info_df, fig = get_model_info(model)
            st.dataframe(info_df)
            st.pyplot(fig=fig)

    with st.expander("Test data", icon=":material/dataset:"):
        st.dataframe(test_df, hide_index=True, use_container_width=True)

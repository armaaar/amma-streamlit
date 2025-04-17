"""Supabase bucket utils"""

import os
import pandas as pd
import streamlit as st

from supabase import create_client, Client

from src.utils.utils import ROOT_DIR

DATASET_FILE_NAME = "Boiler_emulator_dataset.csv"

@st.cache_resource
def get_supabase_client() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

@st.cache_resource
def get_bucket():
    supabase = get_supabase_client()
    return supabase.storage.from_(os.environ["BUCKET_NAME"])

def get_bucket_file_path(file_name: str) -> str:
    local_path = os.path.join(ROOT_DIR, 'models', file_name)

    if not os.path.isfile(local_path):
        with open(local_path, "wb+") as f:
            f.write(get_bucket().download(file_name))

    return local_path

@st.cache_data
def load_full_dataset() -> pd.DataFrame:
    return pd.read_csv(get_bucket_file_path(DATASET_FILE_NAME))

def upload_file_to_bucket(file_name: str) -> bool:
    get_bucket().upload(
        file=os.path.join(ROOT_DIR, 'models', file_name),
        path=file_name,
        file_options={"cache-control": "86400", "upsert": "false"} # 86400 is 1 day
    )
    return True

"""All DB utilities"""

import streamlit as st

from src.db_models.base_model import db

@st.cache_resource
def connect_db() -> None:
    """Connect to peewee DB"""
    db.connect()

"""Model for ml models table"""

from peewee import CharField, BooleanField

from src.db_models.base_model import BaseModel


class MlModel(BaseModel):
    """Table to store ML models in"""

    pickle_path = CharField()
    is_active = BooleanField()

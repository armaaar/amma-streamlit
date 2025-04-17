"""Model for ml models table"""

from peewee import CharField, BooleanField

from db_models.base_model import BaseModel


class MlModel(BaseModel):
    """Table to store ML models in"""

    model_path = CharField()
    is_active = BooleanField()

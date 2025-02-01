"""Model for predictions table"""

from datetime import datetime
from peewee import CharField, ForeignKeyField, DateField

from src.db_models.base_model import BaseModel
from src.db_models.sample import Sample
from src.db_models.ml_model import MlModel


class Prediction(BaseModel):
    """Table to model result and user feedback"""

    date = DateField(default=datetime.now)
    predicted = CharField()
    feedback = CharField(null=True)
    sample = ForeignKeyField(Sample, backref="predictions")
    model = ForeignKeyField(MlModel, backref="results")

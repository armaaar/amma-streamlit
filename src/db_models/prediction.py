"""Model for predictions table"""

from peewee import CharField, ForeignKeyField

from src.db_models.base_model import BaseModel
from src.db_models.sample import Sample
from src.db_models.ml_model import MlModel


class Prediction(BaseModel):
    """Table to model result and user feedback"""

    predicted = CharField()
    feedback = CharField(null=True)
    sample = ForeignKeyField(Sample, backref="predictions")
    model = ForeignKeyField(MlModel, backref="results")

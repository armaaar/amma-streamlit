"""Model for features table"""

from peewee import IntegerField, FloatField, CharField

from src.db_models.base_model import BaseModel


class TestSample(BaseModel):
    """Table to store features of the ml model"""

    fuel_mdot = IntegerField()
    tair = IntegerField()
    treturn = FloatField()
    tsupply = FloatField()
    water_mdot = FloatField()
    prediction = CharField()

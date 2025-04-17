"""Model for features table"""

from peewee import IntegerField, FloatField, CharField

from src.db_models.base_model import BaseModel


class TestSample(BaseModel):
    """Table to store features of the ml model"""

    Fuel_Mdot = IntegerField()
    Tair = IntegerField()
    Treturn = FloatField()
    Tsupply = FloatField()
    Water_Mdot = FloatField()
    prediction = CharField()

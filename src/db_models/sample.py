"""Model for features table"""

from peewee import IntegerField, FloatField

from db_models.base_model import BaseModel


class Sample(BaseModel):
    """Table to store features of the ml model"""

    Fuel_Mdot = IntegerField()
    Tair = IntegerField()
    Treturn = FloatField()
    Tsupply = FloatField()
    Water_Mdot = FloatField()

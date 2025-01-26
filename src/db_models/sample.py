"""Model for features table"""

from peewee import IntegerField, FloatField

from src.db_models.base_model import BaseModel


class Sample(BaseModel):
    """Table to store features of the ml model"""

    fuel_mdot = IntegerField()
    tair = IntegerField()
    treturn = FloatField()
    tsupply = FloatField()
    water_mdot = FloatField()

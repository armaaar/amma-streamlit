"""Base model for all db models"""

import os

from peewee import PostgresqlDatabase, Model

db = PostgresqlDatabase(
    os.environ["DB_NAME"],
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    host=os.environ["DB_HOST"],
    port=os.environ["DB_PORT"]
)


class BaseModel(Model):
    """Base model that all other db models extend"""

    class Meta:
        """Connect models to the Database"""

        database = db

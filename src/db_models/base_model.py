"""Base model for all db models"""

import os

from peewee import SqliteDatabase, Model

from src.utils.utils import ROOT_DIR

DB_FILE_PATH = os.path.join(ROOT_DIR, "data/database.db")

db = SqliteDatabase(DB_FILE_PATH)


class BaseModel(Model):
    """Base model that all other db models extend"""

    class Meta:
        """Connect models to the Database"""

        database = db

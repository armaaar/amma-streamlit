"""A mock of a real model"""

import random
import pandas as pd


class MockModel:
    """Mocks the interface of a real model"""

    possible_outputs: list = []

    def fit(self, _: list, output: list) -> None:
        """Extract the output array from training data"""
        self.possible_outputs = pd.Series(output).unique().tolist()
        return self

    def predict(self, data: list) -> list:
        """Generate random predictions"""
        predictions = []
        for i in range(len(data)):
            predictions.append(random.choice(self.possible_outputs))
        return predictions

import unittest
from unittest.mock import MagicMock
import tempfile

import pandas as pd

from train.train import run
from preprocessing.preprocessing import utils


def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })

class TestPredict(unittest.TestCase):

    # use the function defined above as a mock for utils.LocalTextCategorizationDataset.load_dataset
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_predict(self):

        # create a dictionary params for train conf
        params ={
                "batch_size": 1,
                "epochs": 1,
                "dense_dim": 64,
                "min_samples_per_label": 4,
                "verbose": 1
            }

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            accuracy, _ = run.train("fake_path", params, model_dir, True)
            textpredictmodel = run.TextPredictionModel.from_artefacts(model_dir)

            #predict
            predictions_obtained = textpredictmodel.predict(['php'], 2)

        
        # assert that the prediction shape is correct
        self.assertGreaterEqual(predictions_obtained.shape, (1, 2))


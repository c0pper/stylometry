import json
import time
import joblib
from pathlib import Path

import pandas as pd

from stylometry_utils.decorators import timeit
import numpy as np

from stylometry_utils.class_Experiment import Experiment
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential


class StyloExperiment(Experiment):
    """
    This class gathers the main common attrs and methods used to carry out experiments using a stylometry based
    dataset. By default, the training uses a keras.Sequential neural network build from a dictionary of neural net
    parameters and a default dictionary of parameters is passed, which is reported here as an example:

    .. code-block:: python

        self.nn_parameters = {
            "n_layers": 1,
            "n_units_input": 51,
            "activation": "relu",
            "n_units_l0": 80,
            "dropout_l0": 0.3203504513234906,
            "learning_rate": 0.0014392587661767942,
            "optimizer": "RMSprop",
            "loss": "binary_crossentropy",
            "metrics": ["accuracy"]
        }
    """
    def __init__(self, dataset_path: str, target_col: str, algo=Sequential, nn_parameters=None):
        super().__init__(dataset_path=dataset_path,
                         target_col=target_col)

        if self.number_of_classes > 2:
            raise NotImplementedError("Training on datasets with more than 2 classes still not implemented")

        else:
            self.algo = algo()
            self.algo_name = type(self.algo).__name__
            self.experiment_name = f"{self.dataset_stem}__{self.algo_name}_stylo__{self.now}"
            self.model_savepath = Path(self.dataset_logs_path / type(self).__name__ / self.experiment_name / "models")
            self.log_folder = Path(self.dataset_logs_path / type(self).__name__) / self.experiment_name / "logs"

            if self.split:
                self.X_train, self.X_test, self.y_train, self.y_test = self.split_dataset()

            if not nn_parameters:
                self.nn_parameters = {
                    "n_layers": 1,
                    "n_units_input": 51,
                    "activation": "relu",
                    "n_units_l0": 80,
                    "dropout_l0": 0.3203504513234906,
                    "learning_rate": 0.0014392587661767942,
                    "optimizer": "RMSprop",
                    "loss": "binary_crossentropy",
                    "metrics": ["accuracy"]
                }

    @timeit
    def _train(self, epochs: int):
        if not self.split:
            raise ValueError("Experiment split attribute is not set to true. Can't train with unsplitted dataset.\
             Set the experiment split attr to true.")
        else:
            self.algo.add(
                Dense(
                    self.nn_parameters["n_units_input"],
                    input_dim=self.X_train.shape[1],
                    activation=self.nn_parameters["activation"],
                )
            )
            for i in range(self.nn_parameters["n_layers"]):
                self.algo.add(
                    Dense(
                        self.nn_parameters[f"n_units_l{i}"],
                        activation=self.nn_parameters["activation"],
                    )
                )
                self.algo.add(
                    Dropout(self.nn_parameters[f"dropout_l{i}"])
                )
            self.algo.add(Dense(1, activation="sigmoid"))

            # We compile our model with a sampled learning rate.
            learning_rate = self.nn_parameters["learning_rate"]
            optimizer_name = self.nn_parameters["optimizer"]
            self.algo.compile(
                loss=self.nn_parameters["loss"],
                optimizer=getattr(keras.optimizers, optimizer_name)(learning_rate=learning_rate),
                metrics=self.nn_parameters["metrics"],
            )

            history = self.algo.fit(
                self.X_train,
                self.y_train,
                batch_size=512,
                epochs=epochs,
                validation_data=(self.X_test, self.y_test)
            )

            predicted = self.algo.predict(self.X_test)
            predicted = np.array(list(round(i[0]) for i in predicted))
            report = super().print_report(predicted, self.y_test, target_names=self.lbl_enc.inverse_transform(list(set(self.y_test))))
            self.save_model(self.algo, self.scaler, self.lbl_enc)

            return {
                "history": history.history,
                "report": report,
                "epochs": epochs
            }

    def train(self, epochs: int = 10):
        """
        Trains the :attr:`StyloExperiment.algo` algorithm with the train set originated from the
        :obj:`split_dataset <stylometry_utils.class_Experiment.Experiment.split_dataset>` method.
        Uses a neural net built from the :attr:`StyloExperiment.nn_parameters` dictionary, using the following default
        one if None is passed::

            self.nn_parameters = {
                "n_layers": 1,
                "n_units_input": 51,
                "activation": "relu",
                "n_units_l0": 80,
                "dropout_l0": 0.3203504513234906,
                "learning_rate": 0.0014392587661767942,
                "optimizer": "RMSprop",
                "loss": "binary_crossentropy",
                "metrics": ["accuracy"]
            }

        :param epochs: Epochs to be used in training
        :return: dictionary with the logs of the training
        """
        results, elapsed = self._train(epochs)
        log_dict = self.log(results["history"], results["report"], results["epochs"],
                            len(self.X_train) + len(self.X_test), elapsed)

        return log_dict

    def save_model(self, model, scaler, lbl_enc):
        """
        Saves the model resulting from the training.

        :param model: Trained algorithm
        :param scaler: Scaler fitted during training
        :param lbl_enc: LabelEncoder fitted during the training
        """
        save_confirmation = input(f"Save model {self.experiment_name}? (y/n)")
        if save_confirmation == "y":
            self.model_savepath.mkdir(exist_ok=True, parents=True)
            model_path = Path(self.model_savepath / f"{self.experiment_name}.pkl")
            print(f"Saving model to {model_path}")
            data = {
                "model": model,
                "scaler": scaler,
                "lbl_enc": lbl_enc
            }
            joblib.dump(data, model_path)
        else:
            print("Model wasn't saved")

    @staticmethod
    def load_model_and_predict(model_path: str, X: pd.DataFrame):
        """
        Loads a saved model and uses it to make predictions on a pandas Series.

        :param model_path: Path to the model
        :param X: Series on which to predict
        :return: List of predictions
        """
        data = joblib.load(model_path, mmap_mode=None)
        model = data["model"]
        predicted = model.predict(X)
        predicted = np.array(list(round(i[0]) for i in predicted))
        return predicted

    @timeit
    def evaluate_on_other_dataset(self, testdataset_path: str, model_path: str, dropna=False):
        """
        Evaluate model on a third party tests dataset to verify model robustness.

        :param testdataset_path: Path of the third party dataset
        :param model_path: Path to the model
        :param dropna: Whether to drop na values or not
        """
        X, y, _ = self.load_test_dataset(testdataset_path, dropna=dropna)
        X = self.scaler.transform(X)
        target_names = joblib.load(model_path, mmap_mode=None)["lbl_enc"].classes_

        predicted = self.load_model_and_predict(model_path, X)
        self.print_report(predicted, y, target_names)

    def log(self, history: dict, report: dict, epochs: int, dataset_len: int, elapsed: float) -> dict:
        """
        Saves to a json file useful metrics from the training process. Meant to be used by the
        :meth:`StyloExperiment.train` method.

        :param history: history object of the training process
        :param report: Report metrics of the training from :obj:`Experiment.print_report() <stylometry_utils.class_Experiment.Experiment.print_report>` method
        :param epochs: Epochs used by the training
        :param dataset_len: Lenght of the complete dataset (train + test)
        :param elapsed: Elapsed time during training
        :return: dictionary with all the metrics from the training. Same that was saved to file.
        """
        log_dict = {
            "library_used": type(self).__name__,
            "dataset_name": self.dataset_name,
            "dataset_lenght": dataset_len,
            "nn_parameters": self.nn_parameters,
            "elapsed": elapsed,
            "epochs": epochs,
            "metrics_report": report,
            "history": history
        }
        self.log_folder.mkdir(parents=True, exist_ok=True)
        filepath = Path(self.log_folder / f"{self.experiment_name}_log.json")
        with open(filepath, 'w') as fp:
            json.dump(log_dict, fp, indent=4)
            print("Log saved to ", filepath)
        return log_dict

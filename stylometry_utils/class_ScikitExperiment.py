import pandas as pd

from stylometry_utils.class_Experiment import PublicExpertiment
import json
from pathlib import Path
from stylometry_utils.decorators import timeit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import joblib


class ScikitExperiment(PublicExpertiment):
    """
    This class gathers the main common attribs and methods used to carry out experiments using the scikitlearn
    framework.
    """
    def __init__(self, dataset_path: Path, target_col: str, text_col: str, algo, split: bool = True,
                 test_size: float = 0.2, preprocess_dataset=True):
        super().__init__(dataset_path=dataset_path,
                         target_col=target_col,
                         text_col=text_col,
                         split=split,
                         test_size=test_size,
                         preprocess_dataset=preprocess_dataset)
        self.algo = algo
        now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.algo_name = type(self.algo).__name__
        self.experiment_name = f"{self.dataset_stem}__{self.algo_name}__{now}"
        self.model_savepath = Path(self.dataset_logs_path / type(self).__name__ / self.experiment_name / "models")
        self.log_folder = Path(self.dataset_logs_path / type(self).__name__) / self.experiment_name / "logs"

    @timeit
    def _train(self):
        """
        Trains the :attr:`ScikitExperiment.algo` algorithm with the train set originated from the
         :meth:`Experiment.split_dataset` method. Uses a pipeline composed of TFIDF Vectorizer and the chosen algorithm.
        """
        if not self.split:
            raise ValueError("Experiment split attribute is not set to true. Can't train with unsplitted dataset.\
             Set the experiment split attr to true.")
        else:
            clf_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', self.algo),
            ])
            print(f"Fitting pipeline: {clf_pipeline}")
            clf_pipeline.fit(self.X_train, self.y_train)
            predicted = clf_pipeline.predict(self.X_test)
            report = self.print_report(predicted, self.y_test,
                                       target_names=self.lbl_enc.inverse_transform(list(set(self.y_test))))

            self.save_model(clf_pipeline, self.lbl_enc)

            return report

    def train(self) -> dict:
        """
        Trains the :attr:`ScikitExperiment.algo` algorithm with the train set originated from the
        :meth:`Experiment.split_dataset` method. Uses a pipeline composed of TFIDF Vectorizer and the chosen algorithm.

        :return: dictionary with the logs of the training
        """
        report, elapsed = self._train()
        log_dict = self.log(len(self.X_train) + len(self.X_test), elapsed, report)
        return log_dict

    def save_model(self, model, lbl_enc):
        """
        Saves the model resulting from the training.

        :param model: trained algorithm
        :param lbl_enc: LabelEncoder fitted during the training
        """
        save_confirmation = input(f"Save model {self.experiment_name}? (y/n)")
        if save_confirmation == "y":
            self.model_savepath.mkdir(exist_ok=True, parents=True)
            model_path = Path(self.model_savepath / f"{self.experiment_name}.pkl")
            print(f"Saving model to {model_path}")
            data = {
                "model": model,
                "lbl_enc": lbl_enc
            }
            joblib.dump(data, model_path)
        else:
            print("Model wasn't saved")

    @staticmethod
    def load_model_and_predict(model_path: str, X: pd.Series) -> pd.Series:
        """
        Loads a saved model and uses it to make predictions on a pandas Series.

        :param model_path: Path to the model
        :param X: Series on which to predict
        :return: List of predictions
        """
        data = joblib.load(model_path, mmap_mode=None)
        model = data["model"]  # sklearn
        predicted = model.predict(X)
        return predicted

    @timeit
    def evaluate_on_other_dataset(self, testdataset_path: str, model_path: str, text_col: str, dropna=False):
        """
        Evaluate model on a third party tests dataset to verify model robustness.

        :param testdataset_path: Path of the third party dataset
        :param model_path: Path to the model
        :param text_col: Label of the text column in the third party dataset
        :param dropna: Whether to drop na values or not
        """
        X, y, _ = self.load_test_dataset(testdataset_path, dropna=dropna)
        X = X[text_col]
        target_names = joblib.load(model_path, mmap_mode=None)["lbl_enc"].classes_

        predicted = self.load_model_and_predict(model_path, X)
        super().print_report(predicted, y, target_names)

    def log(self, dataset_len, elapsed, report):
        """
        Saves to a json file useful metrics from the training process. Meant to be used by the
        :meth:`ScikitExperiment.train` method.

        :param dataset_len: Lenght of the dataset
        :param elapsed: Elapsed time during training
        :param report: Report metrics of the training from :meth:`Experiment.print_report` method
        :return: dictionary with all the metrics from the training. Same that was saved to file.
        """
        log_dict = {
            "library_used": type(self).__name__,
            "dataset_name": self.dataset_name,
            "dataset_lenght": dataset_len,
            "algo": self.algo_name,
            "elapsed": elapsed,
            "metrics_report": report
        }
        self.log_folder.mkdir(parents=True, exist_ok=True)
        filepath = Path(self.log_folder / f"{self.experiment_name}_log.json")
        with open(filepath, 'w') as fp:
            json.dump(log_dict, fp)
            print("Log saved to ", filepath)
        return log_dict

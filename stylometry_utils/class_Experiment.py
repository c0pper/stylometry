from pathlib import Path
import os
import inspect

import nltk.stem
from requests import get
from typing import Tuple, Union
import re

import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

MODEL_SAVEPATH = "/content/drive/MyDrive/simo/"


class Experiment:
    """
    A generic experiment class with everything that's needed by more specific types of experiment to subclass off of.
    """
    def __init__(self, dataset_path: Path, target_col: str, split: bool = True, test_size: float = 0.2,
                 model_savepath=MODEL_SAVEPATH):

        self.scaler = preprocessing.StandardScaler()
        self.lbl_enc = preprocessing.LabelEncoder()

        self.dataset_path = dataset_path
        self.test_size = test_size
        self.target_col = target_col
        self.dataset_name = dataset_path.name
        self.model_savepath = model_savepath
        self.logs_path = self._create_log_folder()

        self.X, self.y = self.load_dataset(self.dataset_path)
        if split:
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_dataset()

    def _create_log_folder(self, colab_path: str = r"/content/drive/MyDrive/stylo_experiments/logs"):
        """
        Creates a generic log folder (if it doesn't exist) in the same folder of the passed dataset. Also creates a
        subfolder with the name of the dataset. More specific experiments will create subfolders with their type names
        (e.g. logs/dataset1/[sklearn, bert, stylo])

        :param colab_path: default path of the logs folder if in colab, can be changed or left as is
        :return: path of the logs folder
        """
        if os.getcwd() == "/content":
            mount_path = Path('/content/drive')
            if mount_path.exists():
                path = Path(Path(colab_path) / self.dataset_path.stem)
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Can't find {mount_path}. Please mount your drive using\
                 drive.mount('/content/drive')")
        else:
            path = Path(self.dataset_path.parent / "logs" / f"{self.dataset_path.stem}")
            path.mkdir(parents=True, exist_ok=True)

        return path

    def load_dataset(self, dataset_path: Path) -> Tuple:
        """
        Loads in memory a csv or xlsx or xls dataset.

        :param dataset_path: Path of the dataset
        :return: X and y tuple (y is determined by the :attr:`target_col` attr)
        """
        file_format = dataset_path.suffix
        valid = {".csv", ".xlsx", ".xls"}
        if file_format not in valid:
            raise ValueError(f"results: status must be one of {valid}.")
        else:
            print(f"Loading dataset {dataset_path.name}")
            if file_format == ".csv":
                dataset = pd.read_csv(dataset_path)
                print("Dataset head\n")
                print(dataset.head())
            elif file_format == ".xlsx" or file_format == ".xls":
                dataset = pd.read_excel(dataset_path)
                print("Dataset head\n")
                print(dataset.head())

            X = dataset.drop(self.target_col, axis=1)
            y = self.lbl_enc.fit_transform(dataset[self.target_col].values)

        return X, y

    def split_dataset(self, test_size: float = None) -> Tuple:
        """
        Split the dataset using sklearn train_test_split

        :param test_size: size in decimal of the holdout data
        :return: tuple of X_train, X_test, y_train, y_test
        """
        if not test_size:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=42,
                                                                test_size=self.test_size, shuffle=True)
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=42, test_size=test_size,
                                                                shuffle=True)
            return X_train, X_test, y_train, y_test

    @staticmethod
    def drop_na(X: pd.DataFrame, how: str = "any", axis: int = 0) -> pd.DataFrame:
        """
        Drops nan rows (default) or columns using pandas dropna method.

        :param X: dataframe without target column. Returned by :meth:`Experiment.load_dataset`
        :param how: Determine if row or column is removed from DataFrame, when we have at least one NA or all NA.
        :param axis: Determine if rows or columns which contain missing values are removed. *0, or ‘index’ : Drop rows which contain missing values. *1, or ‘columns’ : Drop columns which contain missing value.
        :return: X with dropped nans
        """
        print("DROPPING NAN")
        X_drop = X.dropna(axis=axis, how=how)

        return X_drop

    def use_scaler(self) -> pd.DataFrame:
        """
        Use standard scaler from sklearn on X.

        :return: scaled X
        """
        X = self.scaler.fit_transform(self.X)
        return X

    @staticmethod
    def preprocess(text: str, stem=False, language: str = "english", stemmer=SnowballStemmer,
                   lemmatizer=WordNetLemmatizer) -> str:
        """
        Preprocess text rows of the dataset before transforming the text into vectors/matrix. Replaces multiple symbols
        with just one, double spaces, links, emojis.

        :param text: Text to be processed
        :param stem: True will stem, False will lemmatize
        :param language: Language to be used by stemmer/lemmatizer
        :param stemmer: Stemmer to be used
        :param lemmatizer: Lemmatizer to be used
        :return: Processed string
        """
        stop_words = stopwords.words('english')

        text = text.lower()  # lowercase

        text = re.sub(r'!+', '!', text)
        text = re.sub(r'\?+', '?', text)
        text = re.sub(r'\.+', '..', text)
        text = re.sub(r"'", "", text)
        text = re.sub(' +', ' ', text).strip()  # Remove and double spaces
        text = re.sub(r'&amp;?', r'and', text)  # replace & -> and
        text = re.sub(r"https?://t.co/[A-Za-z0-9]+", "", text)  # Remove URLs
        # remove some puncts (except . ! # ?)
        text = re.sub(r'[:"$%&\*+,-/;<=>@\\^_`{|}~]+', '', text)
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'EMOJI', text)

        tokens = []
        if not stem:
            lemmatizer = lemmatizer()
            for token in text.split():
                if token not in stop_words:
                    tokens.append(lemmatizer.lemmatize(token))
            return " ".join(tokens)
        else:
            stemmer = stemmer(language)
            for token in text.split():
                if token not in stop_words:
                    tokens.append(stemmer.stem(token))
            return " ".join(tokens)

    def show_class_distribution(self):
        """
        Show how many instances of each class are in the dataset

        """
        listy = list(self.lbl_enc.inverse_transform(self.y))
        print("Dataset class distribution:")
        for i in set(listy):
            print(i, listy.count(i))

    @staticmethod
    def print_cm(yvalid: Union[pd.Series, list], predicted: Union[pd.Series, list], target_names=None):
        """
        Shows confusion matrix.

        :param yvalid: Ground truth labels
        :param predicted: predicted labels
        :param target_names: list of class labels
        """
        if not target_names:
            target_names = []
        cm = metrics.confusion_matrix(yvalid, predicted)
        disp = metrics.ConfusionMatrixDisplay(cm, display_labels=target_names)
        disp.plot(xticks_rotation="vertical")

    def print_report(self, predicted, yvalid, target_names=None):
        """
        Shows skelarn report.

        :param predicted: predicted labels
        :param yvalid: Ground truth labels
        :param target_names: list of class labels
        :return:
        """
        report_dict = metrics.classification_report(yvalid, predicted, target_names=[str(x) for x in target_names],
                                                    output_dict=True)
        report_text = metrics.classification_report(yvalid, predicted, target_names=[str(x) for x in target_names])
        print(report_text)
        self.print_cm(yvalid, predicted, target_names=target_names)
        return report_dict

    def load_test_dataset(self, testdataset_path: Path, dropna: bool = False, how: str = "any", axis: int = 0) -> Tuple:
        """
        Load a third party tests dataset to be used for verifying model robustness.

        :param testdataset_path: path of the tests dataset
        :param dropna: Whether to drop na values or not
        :param how: Determine if row or column is removed from DataFrame, when we have at least one NA or all NA.
        :param axis:Determine if rows or columns which contain missing values are removed. *0, or ‘index’ : Drop rows which contain missing values. *1, or ‘columns’ : Drop columns which contain missing value.
        :return:
        """
        X, y = self.load_dataset(testdataset_path)
        target_names = self.lbl_enc.inverse_transform(list(set(y)))
        if dropna:
            self.drop_na(X)

        return X, y, target_names

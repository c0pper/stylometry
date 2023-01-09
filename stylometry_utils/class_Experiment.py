from pathlib import Path
import os
from tqdm import tqdm
from stylometry_utils.decorators import timeit

from typing import Tuple, Union, Callable
import re

import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

MODEL_SAVEPATH = Path("/content/drive/MyDrive/simo/")
tqdm.pandas()


class Experiment:
    """
    A generic experiment class with everything that's needed by more specific types of experiment to subclass off of.
    """

    def __init__(self, dataset_path: Path, target_col: str, split: bool = True, test_size: float = 0.2,
                 model_savepath: Path = MODEL_SAVEPATH):

        self.scaler = preprocessing.StandardScaler()
        self.lbl_enc = preprocessing.LabelEncoder()

        self.dataset_path = dataset_path
        self.dataset_folder = dataset_path.parent
        self.dataset_name = dataset_path.name
        self.dataset_stem = dataset_path.stem
        self.test_size = test_size
        self.target_col = target_col
        self.split = split
        self.model_savepath = model_savepath
        self.dataset_logs_path = self._create_log_folder()

        self.preprocessed_file = Path(self.dataset_folder / f"{self.dataset_stem}_preprocessed.csv")
        if Path(self.dataset_folder / f"{self.dataset_stem}_preprocessed.csv").exists():
            print(f"Found preprocessed dataset at {self.preprocessed_file}, loading already preprocessed file and \
            skipping preprocessing")
            self.X, self.y = self.load_dataset(self.preprocessed_file, self.target_col)
        else:
            self.X, self.y = self.load_dataset(self.dataset_path, self.target_col)

    def _create_log_folder(self, colab_path: str = r"/content/drive/MyDrive/stylo_experiments/logs"):
        """
        Creates a generic log folder (if it doesn't exist) in the same folder of the passed dataset or at the
         :attr:`colab_path` argument. Also creates a subfolder with the name of the dataset. More specific experiments
         will create subfolders with their type names (e.g. logs/dataset1/[sklearn, bert, stylo])

        :param colab_path: default path of the logs folder if in colab, can be changed or left as is
        :return: path of the logs folder
        """
        if os.getcwd() == "/content":
            mount_path = Path('/content/drive')
            if mount_path.exists():
                path = Path(Path(colab_path) / self.dataset_stem)
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Can't find {mount_path}. Please mount your drive using\
                 drive.mount('/content/drive')")
        else:
            path = Path(self.dataset_folder / "logs" / f"{self.dataset_stem}")
            path.mkdir(parents=True, exist_ok=True)

        return path

    def load_dataset(self, dataset_path: Path, target_col: str = "label") -> Tuple:
        """
        Loads in memory a csv or xlsx or xls dataset.

        :param target_col: label of the target column in the dataset
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

            X = dataset.drop(target_col, axis=1)
            y = self.lbl_enc.fit_transform(dataset[target_col].values)

        return X, y

    def split_dataset(self, test_size: float = None) -> Tuple:
        """
        Split the dataset using sklearn train_test_split

        :param test_size: size in decimal of the holdout data
        :return: tuple of X_train, X_test, y_train, y_test
        """
        if not test_size:
            test_size = self.test_size
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=42,
                                                            test_size=test_size, shuffle=True)
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
        if target_names is None:
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


class PublicExpertiment(Experiment):
    """
    This class gathers the main common attribs and methods used to carry out experiments using publicly available
    frameworks such as scikitlearn or tensorflow.
    """
    def __init__(self, dataset_path: Path, target_col: str, text_col: str, split: bool = True, test_size: float = 0.2,
                 model_savepath: Path = MODEL_SAVEPATH, preprocess_dataset=True):
        super().__init__(dataset_path=dataset_path,
                         target_col=target_col,
                         split=split,
                         test_size=test_size,
                         model_savepath=model_savepath)
        self.text_col = text_col
        self.preprocess_dataset = preprocess_dataset
        self.X = self.X[self.text_col]
        if self.preprocess_dataset:
            if not self.preprocessed_file.exists():
                self.X, self.preprocessing_time = self.apply_preprocess(self.X, self.preprocess)

        if self.split:
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_dataset()

    @timeit
    def apply_preprocess(self, text_series: pd.Series, preprocessing_function: Callable,
                         save_encoding: str = "utf-8-sig") -> pd.Series:
        """
        Preprocesses the text column of the dataset using the passed preprocessing function. Preprocessing function
        must accept a string. Processed series is concatenated with :attr:`Experiment.y` and the resulting dataframe is
        saved in the :attr:`Experiment.dataset_folder` folder to be retrieved in future experiments and save time on
        preprocessing.

        :param text_series: pandas Series of the text rows in the dataset
        :param save_encoding: the encoding to be used by the :meth:`Experiment._save_preprocessed_dataset` method
        :param preprocessing_function: Function to be used in preprocessing. Default is :meth:`Experiment.preprocess`
        :return: returns Tuple (processed series: Series, elapsed seconds: float)
        """
        print("\nPreprocessing texts...")
        print(f"\nBefore: {text_series}")
        text_series_processed = text_series.progress_apply(lambda x: preprocessing_function(x))
        print(f"\nAfter: {text_series_processed}")
        self._save_preprocessed_dataset(text_series_processed, encoding=save_encoding)

        return text_series_processed

    def _save_preprocessed_dataset(self, processed_series: pd.Series, encoding: str = "utf-8-sig"):
        """
        Saves a preprocessed dataset in order to avoid preprocessing it everytime it is loaded. Meant to be used by
        the :meth;`PublicExperiment.apply_preprocess` method, not manually.

        :param processed_series: output of
        :param encoding: encoding to be used in saving the csv file
        """
        frame = {self.text_col: processed_series, self.target_col: self.y}
        processed_ds = pd.DataFrame(frame)
        processed_ds.to_csv(self.dataset_folder / f"{self.dataset_stem}_preprocessed.csv", index=False,
                            encoding=encoding)

    def load_test_dataset(self, testdataset_path: str, target_col: str = "label", dropna: bool = False,
                          how: str = "any", axis: int = 0) -> Tuple:
        """
        Load a third party tests dataset to be used for verifying model robustness.

        :param target_col: label of the target column in the dataset
        :param testdataset_path: path of the test dataset
        :param dropna: Whether to drop na values or not
        :param how: Determine if row or column is removed from DataFrame, when we have at least one NA or all NA.
        :param axis: Determine if rows or columns which contain missing values are removed. *0, or ‘index’ : Drop rows which contain missing values. *1, or ‘columns’ : Drop columns which contain missing value.
        :return: Tuple of X, y and list of target names
        """
        testdataset_path = Path(testdataset_path)
        preprocessed_test_dataset_path = Path(testdataset_path.parent / f"{testdataset_path.stem}_preprocessed.csv")
        if preprocessed_test_dataset_path.exists():
            print(f"Found preprocessed dataset at {preprocessed_test_dataset_path}, loading already preprocessed file \
            and skipping preprocessing")
            X, y = self.load_dataset(preprocessed_test_dataset_path, target_col)
        else:
            X, y = self.load_dataset(preprocessed_test_dataset_path, target_col)
        target_names = self.lbl_enc.inverse_transform(list(set(y)))
        if dropna:
            self.drop_na(X, how=how, axis=axis)

        return X, y, target_names


# e = PublicExpertiment(Path(r"C:\Users\smarotta\PycharmProjects\stylometry\fn20k.xlsx"), target_col="label",
#                       text_col="text")
# e.X = e.apply_preprocess(e.preprocess)
# test = e.apply_preprocess(e.preprocess)

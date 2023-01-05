from pathlib import Path
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from datetime import datetime
import joblib
from tqdm import tqdm
from transformers import TFBertModel, BertTokenizer
import numpy as np
from tensorflow import keras
from requests import get
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import time
import json
import os
from tqdm import tqdm
from typing import Tuple
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Dense, Dropout

tqdm.pandas()
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')

MODEL_SAVEPATH = "/content/drive/MyDrive/simo/"
TIMENOW = datetime.now().strftime('%d-%m-%y-%H-%M')

os.makedirs("/content/drive/MyDrive/simo/logs", exist_ok=True)
os.makedirs("/content/drive/MyDrive/simo/logs/sklearn", exist_ok=True)
os.makedirs("/content/drive/MyDrive/simo/logs/bert", exist_ok=True)
os.makedirs("/content/drive/MyDrive/simo/logs/stylo", exist_ok=True)
os.makedirs("/content/drive/MyDrive/simo/models", exist_ok=True)
os.makedirs("/content/drive/MyDrive/simo/models/sklearn", exist_ok=True)
os.makedirs("/content/drive/MyDrive/simo/models/bert", exist_ok=True)
os.makedirs("/content/drive/MyDrive/simo/models/stylo", exist_ok=True)


def preprocess(text, stem=False):
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
        lemmatizer = WordNetLemmatizer()
        for token in text.split():
            if token not in stop_words:
                tokens.append(lemmatizer.lemmatize(token))
        return " ".join(tokens)
    else:
        stemmer = SnowballStemmer('english')
        for token in text.split():
            if token not in stop_words:
                tokens.append(stemmer.stem(token))
        return " ".join(tokens)


def apply_preprocess(xtrain, xvalid):
    print("\nPreprocessing texts...")
    print(f"\nBefore: {(xtrain.iloc[0][:50] + '..') if len(xtrain.iloc[0]) > 50 else xtrain.iloc[0]}")
    xtrain = xtrain.progress_apply(lambda x: preprocess(x))
    xvalid = xvalid.progress_apply(lambda x: preprocess(x))
    print(f"\nAfter: {(xtrain.iloc[0][:50] + '..') if len(xtrain.iloc[0]) > 50 else xtrain.iloc[0]}")

    return xtrain, xvalid


class Experiment:

    def __init__(self, dataset_path: Path, target_col: str, split_size: float = 0.8, model_savepath=MODEL_SAVEPATH):
        self.scaler = StandardScaler()
        self.lbl_enc = preprocessing.LabelEncoder()

        self.dataset_path = dataset_path
        self.X = self.load_dataset(self.dataset_path)[0]
        self.y = self.load_dataset(self.dataset_path)[1]
        self.split_size = split_size
        self.target_col = target_col
        self.dataset_name = dataset_path.split(".")[-2].split("/")[-1]
        self.model_savepath = model_savepath

        # check if on colab or local
        if os.getcwd() == "/content":
            self.nb_name = get('http://172.28.0.2:9000/api/sessions').json()[0]['name'].split(".")[0]
        else:
            self.nb_name = Path(__file__).name

    def load_dataset(self, dataset_path: Path) -> Tuple:
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

    def split_dataset(self) -> Tuple:
        xtrain, xvalid, ytrain, yvalid = train_test_split(self.X, self.y, random_state=42, test_size=self.split_size,
                                                          shuffle=True)
        return xtrain, xvalid, ytrain, yvalid

    def drop_na(self, how: str = "any") -> pd.DataFrame:
        print("DROPPING NAN")
        X = self.X.dropna(axis=0, how=how)

        return X

    def use_scaler(self) -> pd.DataFrame:
        X = self.scaler.fit_transform(self.X)
        return X

    def show_class_distribution(self):
        listy = list(self.lbl_enc.inverse_transform(self.y))
        print("Dataset class distribution:")
        for i in set(listy):
            print(i, listy.count(i))

    def load_split_dataset(self, dataset_path, dropna=False, do_split=True, use_scaler=False):

        file_format = dataset_path.split(".")[-1]
        valid = {"csv", "xlsx", "xls"}
        if file_format not in valid:
            raise ValueError(f"results: status must be one of {valid}.")
        else:
            if file_format == "csv":
                dataset = pd.read_csv(dataset_path)
                print("Dataset head\n")
                print(dataset.head())
            elif file_format == "xlsx" or file_format == "xls":
                dataset = pd.read_excel(dataset_path)
                print("Dataset head\n")
                print(dataset.head())

        X = dataset.drop(self.target_col, axis=1)
        y = self.lbl_enc.fit_transform(dataset[self.target_col].values)

        # dropping nans
        if dropna:
            print("DROPPING NAN")
            dataset = dataset.dropna(axis=0, how='any')

        if use_scaler:
            X = Experiment.scaler.fit_transform(X)

        listy = list(self.lbl_enc.inverse_transform(y))
        print("Dataset class distribution:")
        for i in set(listy):
            print(i, listy.count(i))

        if do_split:
            xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=self.split_size,
                                                              shuffle=True)
            return xtrain, xvalid, ytrain, yvalid
        else:
            return X, y

    def print_cm(self, yvalid, predicted, target_names=[]):
        cm = metrics.confusion_matrix(yvalid, predicted)
        disp = metrics.ConfusionMatrixDisplay(cm, display_labels=target_names)
        disp.plot(xticks_rotation="vertical")

    def print_report(self, predicted, yvalid, target_names=None):
        report_dict = metrics.classification_report(yvalid, predicted, target_names=[str(x) for x in target_names],
                                                    output_dict=True)
        report_text = metrics.classification_report(yvalid, predicted, target_names=[str(x) for x in target_names])
        print(report_text)
        self.print_cm(yvalid, predicted, target_names=target_names)
        return report_dict

    def load_test_dataset(self, testdataset_path, dropna):
        X, y = self.load_split_dataset(testdataset_path, dropna=dropna, do_split=False)
        target_names = self.lbl_enc.inverse_transform(list(set(y)))
        return X, y, target_names


class PublicExpertiment(Experiment):
    def __init__(self, dataset_path, split_size, target_col, text_col, model_savepath=MODEL_SAVEPATH,
                 preprocess_dataset=True):
        super().__init__(dataset_path, split_size, target_col, model_savepath)
        self.text_col = text_col
        self.preprocess_dataset = preprocess_dataset


class ScikitExperiment(PublicExpertiment):
    def __init__(self, dataset_path, split_size, target_col, text_col, algo, model_savepath=MODEL_SAVEPATH,
                 preprocess_dataset=True):
        super().__init__(dataset_path=dataset_path,
                         split_size=split_size,
                         target_col=target_col,
                         text_col=text_col,
                         model_savepath=model_savepath,
                         preprocess_dataset=preprocess_dataset
                         )
        self.algo = algo

    def train(self, dropna=False):
        start = time.time()
        xtrain, xvalid, ytrain, yvalid = super().load_split_dataset(self.dataset_path, dropna=dropna)
        xtrain = xtrain[self.text_col]
        xvalid = xvalid[self.text_col]

        if self.preprocess_dataset:
            xtrain, xvalid = apply_preprocess(xtrain, xvalid)

        clf_pipeline = Pipeline([
            ('ctv', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', self.algo),
        ])
        print(f"Fitting pipeline: {clf_pipeline}")
        clf_pipeline.fit(xtrain, ytrain)
        end = time.time()
        elapsed = round(end - start, 2)
        predicted = clf_pipeline.predict(xvalid)
        report = super().print_report(predicted, yvalid, target_names=self.lbl_enc.inverse_transform(list(set(yvalid))))

        self.save_model(clf_pipeline, self.lbl_enc)
        print("Time elapsed in seconds: ", round(elapsed, 2))

        log_dict = self.log(self.model_savepath, self.dataset_path.split("/")[-1], len(xtrain) + len(xvalid),
                            type(self.algo).__name__, elapsed, report)
        return log_dict

    def save_model(self, model, lbl_enc):
        algo_name = type(self.algo).__name__
        experiment_name = self.nb_name + "__" + self.dataset_name + "__" + algo_name
        save_confirmation = input(f"Save model {experiment_name}? (y/n)")
        if save_confirmation == "y":
            filepath = f'{self.model_savepath}models/sklearn/{experiment_name}.pkl'
            print(f"Saving model to {filepath}")
            data = {
                "model": model,
                "lbl_enc": lbl_enc
            }
            joblib.dump(data, filepath)
        else:
            print("Model wasn't saved")

    def load_model_and_predict(self, modelpath, X):
        data = joblib.load(modelpath, mmap_mode=None)
        model = data["model"]  # sklearn
        predicted = model.predict(X)
        return predicted

    def evaluate_on_other_dataset(self, testdataset_path, modelpath, text_col, dropna=False):
        start = time.time()
        X, y, _ = super().load_test_dataset(testdataset_path, dropna=dropna)
        X = X[text_col]
        target_names = joblib.load(modelpath, mmap_mode=None)["lbl_enc"].classes_

        predicted = self.load_model_and_predict(modelpath, X)
        super().print_report(predicted, y, target_names)
        end = time.time()
        elapsed = round(end - start, 2)
        print("Time elapsed in seconds: ", round(elapsed, 2))

    def log(self, savepath, datasetname, dataset_len, algo, elapsed, report):
        log_dict = {
            "library_used": type(self).__name__,
            "dataset_name": datasetname,
            "dataset_lenght": dataset_len,
            "algo": algo,
            "elapsed": elapsed,
            "metrics_report": report
        }
        algo_name = type(self.algo).__name__
        experiment_name = self.nb_name + "__" + self.dataset_name + "__" + algo_name
        filepath = f'{savepath}logs/sklearn/{experiment_name}_log.json'
        with open(filepath, 'w') as fp:
            json.dump(log_dict, fp)
            print("Log saved to ", filepath)
        return log_dict


class TFExperiment(PublicExpertiment):
    def __init__(self, dataset_path, split_size, text_col, target_col, preprocess_dataset=True,
                 model_savepath=MODEL_SAVEPATH, bert_pretrained_model='bert-large-uncased', bert_encode_maxlen=60):
        super().__init__(dataset_path=dataset_path,
                         split_size=split_size,
                         text_col=text_col,
                         target_col=target_col,
                         preprocess_dataset=preprocess_dataset,
                         model_savepath=model_savepath)
        self.bert_pretrained_model = bert_pretrained_model
        self.bert_encode_maxlen = bert_encode_maxlen

    def bert_encode(self, data, max_len):
        bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        input_ids = []
        attention_masks = []

        for i in tqdm(range(len(data))):
            encoded = bert_tokenizer.encode_plus(data.iloc[i],
                                                 add_special_tokens=True,
                                                 max_length=max_len,
                                                 pad_to_max_length=True,
                                                 return_attention_mask=True)

            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

        return np.array(input_ids), np.array(attention_masks)

    def create_model(self, bert_encode_maxlen, bert_pretrained_model, optimizer, loss, metrics):
        bert_layers = TFBertModel.from_pretrained(bert_pretrained_model)

        input_ids = keras.Input(shape=(bert_encode_maxlen,), dtype='int32', name='input_ids')
        attention_masks = keras.Input(shape=(bert_encode_maxlen,), dtype='int32', name='attention_masks')

        output = bert_layers([input_ids, attention_masks])
        output = output[1]
        net = keras.layers.Dense(32, activation='relu')(output)
        net = keras.layers.Dropout(0.2)(net)
        net = keras.layers.Dense(1, activation='sigmoid')(net)
        outputs = net
        model = keras.models.Model(inputs=[input_ids, attention_masks], outputs=outputs)

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[metrics])

        model.summary()
        return model

    def train(self, bert_encode_maxlen=None, bert_pretrained_model=None, dropna=False, epochs=10,
              optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics='accuracy',
              callbacks=[]):
        start = time.time()
        if bert_encode_maxlen is None:
            bert_encode_maxlen = self.bert_encode_maxlen
        if bert_pretrained_model is None:
            bert_pretrained_model = self.bert_pretrained_model

        xtrain, xvalid, ytrain, yvalid = super().load_split_dataset(self.dataset_path, dropna=dropna)
        xtrain = xtrain[self.text_col]
        xvalid = xvalid[self.text_col]

        if self.preprocess_dataset:
            xtrain, xvalid = apply_preprocess(xtrain, xvalid)

        train_input_ids, train_attention_masks = self.bert_encode(xtrain, bert_encode_maxlen)
        val_input_ids, val_attention_masks = self.bert_encode(xvalid, bert_encode_maxlen)

        model = self.create_model(bert_encode_maxlen, bert_pretrained_model, optimizer=optimizer, loss=loss,
                                  metrics=metrics)

        history = model.fit(
            [train_input_ids, train_attention_masks],
            ytrain,
            epochs=epochs,
            # validation_data=([val_input_ids, val_attention_masks], y_val),
            batch_size=32,
            # callbacks=callbacks for now no callbacls
        )
        end = time.time()
        elapsed = round(end - start, 2)

        predicted = model.predict([val_input_ids, val_attention_masks])
        predicted = np.array(list(round(i[0]) for i in predicted))
        report = super().print_report(predicted, yvalid, target_names=self.lbl_enc.inverse_transform(list(set(yvalid))))

        self.save_model(model, self.lbl_enc)
        print("Time elapsed in seconds: ", round(elapsed, 2))

        log_dict = self.log(self.model_savepath, self.dataset_path.split("/")[-1], model, len(xtrain) + len(xvalid),
                            elapsed, bert_encode_maxlen, epochs, bert_pretrained_model, optimizer, report)

    def save_model(self, model, lbl_enc):
        algo_name = type(model).__name__
        experiment_name = self.nb_name + "__" + self.dataset_name + "__" + algo_name
        save_confirmation = input(f"Save model {experiment_name}? (y/n)")
        if save_confirmation == "y":
            filepath = f'{self.model_savepath}models/bert/{experiment_name}.h5'
            lbl_enc_path = f'{self.model_savepath}models/bert/{experiment_name}__lbl_enc.pkl'
            data = {
                "model": model,
                "lbl_enc": lbl_enc
            }
            print(f"Saving model to {filepath}")
            # joblib.dump(data, filepath)
            joblib.dump(lbl_enc, lbl_enc_path)
            model.save(f'{self.model_savepath}models/bert/{experiment_name}.h5')
        else:
            print("Model wasn't saved")

    def load_model_and_predict(self, modelpath, X):
        # data = joblib.load(modelpath, mmap_mode=None)
        # model = data["model"]
        model = load_model(modelpath,
                           custom_objects={'TFBertModel': TFBertModel.from_pretrained(self.bert_pretrained_model)})
        predicted = model.predict(X)
        predicted = np.array(list(round(i[0]) for i in predicted))
        return predicted

    def evaluate_on_other_dataset(self, testdataset_path, modelpath, text_col, fitted_lbl_enc, dropna=False):
        start = time.time()
        X, y, _ = super().load_test_dataset(testdataset_path, dropna=dropna)
        target_names = joblib.load(fitted_lbl_enc).classes_
        X = X[text_col]
        input_ids, attention_masks = self.bert_encode(X, self.bert_encode_maxlen)

        predicted = self.load_model_and_predict(modelpath, [input_ids, attention_masks])
        super().print_report(predicted, y, target_names)
        end = time.time()
        elapsed = round(end - start, 2)
        print("Time elapsed in seconds: ", round(elapsed, 2))

    def log(self, savepath, datasetname, model, dataset_len, elapsed, bert_encode_maxlen, epochs, bert_pretrained_model,
            optimizer, report):
        log_dict = {
            "library_used": type(self).__name__,
            "dataset_name": datasetname,
            "dataset_lenght": dataset_len,
            "elapsed": elapsed,
            "bert_encode_maxlen": bert_encode_maxlen,
            "epochs": epochs,
            "bert_pretrained_model": bert_pretrained_model,
            "optimizer": str(optimizer),
            "metrics_report": report
        }
        algo_name = type(model).__name__
        experiment_name = self.nb_name + "__" + self.dataset_name + "__" + algo_name
        filepath = f'{savepath}logs/bert/{experiment_name}_log.json'
        with open(filepath, 'w') as fp:
            json.dump(log_dict, fp)
            print("Log saved to ", filepath)
        return log_dict


class StyloExperiment(Experiment):
    def __init__(self, dataset_path, split_size, target_col, model_savepath=MODEL_SAVEPATH):
        super().__init__(dataset_path=dataset_path,
                         split_size=split_size,
                         target_col=target_col,
                         model_savepath=model_savepath)

    def train(self, epochs=10):
        start = time.time()
        xtrain, xvalid, ytrain, yvalid = super().load_split_dataset(self.dataset_path, use_scaler=True)

        nn_parameters = {
            "n_layers": 1,
            "n_units_input": 51,
            "activation": "relu",
            "n_units_l0": 80,
            "dropout_l0": 0.3203504513234906,
            "learning_rate": 0.0014392587661767942,
            "optimizer": "RMSprop"
        }

        model = keras.models.Sequential()
        model.add(
            Dense(
                nn_parameters["n_units_input"],
                input_dim=xtrain.shape[1],
                activation=nn_parameters["activation"],
            )
        )
        for i in range(nn_parameters["n_layers"]):
            model.add(
                Dense(
                    nn_parameters[f"n_units_l{i}"],
                    activation=nn_parameters["activation"],
                )
            )
            model.add(
                Dropout(nn_parameters[f"dropout_l{i}"])
            )
        model.add(Dense(1, activation="sigmoid"))

        # We compile our model with a sampled learning rate.
        learning_rate = nn_parameters["learning_rate"]
        optimizer_name = nn_parameters["optimizer"]
        model.compile(
            loss="binary_crossentropy",
            optimizer=getattr(keras.optimizers, optimizer_name)(learning_rate=learning_rate),
            metrics=["accuracy"],
        )

        history = model.fit(
            xtrain,
            ytrain,
            batch_size=512,
            epochs=epochs,
            validation_data=(xvalid, yvalid)
        )
        end = time.time()
        elapsed = round(end - start, 2)

        predicted = model.predict(xvalid)
        predicted = np.array(list(round(i[0]) for i in predicted))
        report = super().print_report(predicted, yvalid, target_names=self.lbl_enc.inverse_transform(list(set(yvalid))))
        self.save_model(model, self.scaler, self.lbl_enc)
        print("Time elapsed in seconds: ", round(elapsed, 2))

        log_dict = self.log(self.model_savepath, self.dataset_path.split("/")[-1], model, len(xtrain) + len(xvalid),
                            elapsed, epochs, nn_parameters, report)

    def save_model(self, model, scaler, lbl_enc):
        algo_name = type(model).__name__
        experiment_name = self.nb_name + "__" + self.dataset_name + "__" + algo_name + "_stilometria"
        save_confirmation = input(f"Save model {experiment_name}? (y/n)")
        if save_confirmation == "y":
            filepath = f'{self.model_savepath}models/stylo/{experiment_name}.pkl'
            data = {
                "model": model,
                "scaler": scaler,
                "lbl_enc": lbl_enc
            }
            print(f"Saving model to {filepath}")
            joblib.dump(data, filepath)
            # model.save(f'{self.model_savepath}{experiment_name}.h5')
        else:
            print("Model wasn't saved")

    def load_model_and_predict(self, modelpath, X):
        data = joblib.load(modelpath, mmap_mode=None)
        model = data["model"]
        predicted = model.predict(X)
        predicted = np.array(list(round(i[0]) for i in predicted))
        return predicted

    def evaluate_on_other_dataset(self, testdataset_path, modelpath, dropna=False):
        start = time.time()
        X, y, _ = super().load_test_dataset(testdataset_path, dropna=dropna)
        target_names = joblib.load(modelpath, mmap_mode=None)["lbl_enc"].classes_
        X = self.scaler.transform(X)

        predicted = self.load_model_and_predict(modelpath, X)
        super().print_report(predicted, y, target_names)
        end = time.time()
        elapsed = round(end - start, 2)
        print("Time elapsed in seconds: ", round(elapsed, 2))

    def log(self, savepath, datasetname, model, dataset_len, elapsed, epochs, nn_parameters, report):
        log_dict = {
            "library_used": type(self).__name__,
            "dataset_name": datasetname,
            "dataset_lenght": dataset_len,
            "elapsed": elapsed,
            "epochs": epochs,
            "nueral_net_parameters": nn_parameters,
            "metrics_report": report
        }
        algo_name = type(model).__name__
        experiment_name = self.nb_name + "__" + self.dataset_name + "__" + algo_name + "_stilometria"
        filepath = f'{savepath}logs/stylo/{experiment_name}_log.json'
        with open(filepath, 'w') as fp:
            json.dump(log_dict, fp)
            print("Log saved to ", filepath)
        return log_dict

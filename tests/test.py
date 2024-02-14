from pathlib import Path
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler





# from stylometry_utils.class_ScikitExperiment import ScikitExperiment
# e = ScikitExperiment(Path(r"C:\Users\smarotta\PycharmProjects\stylometry\fn20k.xlsx"), "label", text_col="text", algo=MultinomialNB())
# print(e.number_of_classes)

# from stylometry_utils.class_StyloExperiment import StyloExperiment
# e = StyloExperiment(r"C:\Users\smarotta\PycharmProjects\stylometry\fn20k_stil.csv", "Target")
# print(e.train())
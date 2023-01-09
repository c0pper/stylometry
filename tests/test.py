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

from stylometry_utils.class_ScikitExperiment import ScikitExperiment
e = ScikitExperiment(Path(r"C:\Users\smarotta\PycharmProjects\stylometry\fn20k.xlsx"), "label", text_col="text", algo=MultinomialNB())
e.train()
# print(e.X_train.shape, e.y_train.shape)
# print(e.X_train.shape[0])
# print(e.y_train.shape[0])

# dataset = pd.read_csv(r"C:\Users\smarotta\PycharmProjects\stylometry\fn44k.csv")
# print(dataset)
# lbl_enc = preprocessing.LabelEncoder()
# X = dataset["text"]
# y = lbl_enc.fit_transform(dataset["label"])
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, shuffle=True)





# count_vec = CountVectorizer()
# bow = count_vec.fit_transform(dataset['text'])
# bow = np.array(bow.todense())
# X = bow
# print(X)
# tfidf = TfidfVectorizer()
# X = tfidf.fit_transform(X)
# print(X, y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# # X, y = make_classification(random_state=0)
# model = MultinomialNB()
# model.fit(X_train, y_train)
# print(model.score(X_test,y_test))

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# print(X_train.shape, y_train.shape)
# pipe = Pipeline([('scaler', CountVectorizer()), ('svc', SVC())])
# pipe.fit(X_train, y_train)
# clf_pipeline = Pipeline([
#     ('ctv', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', MultinomialNB()),
# ])
# clf_pipeline.fit(X, y)

# print(X.shape, y.shape)
# model = MultinomialNB().fit(X, y)



# clf_pipeline = Pipeline([
#     ('ctv', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', clf),
# ])
# print(y)
# if X.shape[0] != y.shape[0]:
#     print("X and y rows are mismatched, check dataset again")
# else:
#     print("X and y rows are matched")
# clf_pipeline.fit(X, y)
# from sklearn.naive_bayes import GaussianNB
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y = np.array([1, 1, 1, 2, 2, 2])
# clf = GaussianNB()
# clf.fit(X, Y)
# print(clf.predict([[-0.8, -1]]))
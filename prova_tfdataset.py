import tensorflow as tf
import pandas as pd
import re
from nltk import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords
from sklearn import preprocessing


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


le = preprocessing.LabelEncoder()
df = pd.read_csv(r"C:\Users\smarotta\PycharmProjects\stylometry\fn20k_stil.csv")
target = le.fit_transform(df.pop('Target'))
print(df, target)
df = tf.convert_to_tensor(df)
dataset = tf.data.Dataset.from_tensor_slices(df)
for i in dataset.as_numpy_iterator():
    print(i)
    print("....................")
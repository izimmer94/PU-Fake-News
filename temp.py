from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import nltk
nltk.download('omw-1.4')
print("imports done")
# nltk.download('punkt')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# input_file = "./article_dataset.csv"
# df_raw = pd.read_csv(input_file, index_col="id")
# df_raw = preprocessing.clean_df(df_raw)
# df_raw = df_raw[df_raw['cleaned_text'].str.strip() != ""].copy()
# df_raw.reset_index(inplace=True)
# df_raw['id'] = df_raw.index
# df_raw["clean_title"] = df_raw['title'].apply(preprocessing.clean_text)
# # print(df_raw.head())
# print("df preprocessed")

input_file = "./cleaned_article_dataset.csv"
df = pd.read_csv(input_file, index_col="id")
df = preprocessing.turn_into_pu(df, 0.5)
print("df shape:", df.shape)

X_train, X_test, y_train, y_test = preprocessing.test_train_split(df)
y_train.index = X_train.index
# y_test_true = df_raw.loc[y_test.index, 'label']

X_train['tokens'] = X_train['cleaned_text'].apply(lambda x: x.split())
df.head()

vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=True)
tfidf_matrix = vectorizer.fit_transform(X_train['cleaned_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=X_train.index)

pca = PCA(n_components=6)
principal_components = pca.fit_transform(tfidf_df)
print("Explained Variance: ", pca.explained_variance_ratio_.cumsum())

import pandas as pd
import numpy as np
import preprocessing
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

'''
Logistic Regression Base Classifier
'''
class logisticRegressionCustom:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def _sentiment_make(self, df: pd.DataFrame, type:str):
        df['sentiment'] = df[type].apply(self.analyzer.polarity_scores)
        df = pd.concat([df.drop(['sentiment'], axis=1), df['sentiment'].apply(pd.Series)], axis=1)
        df.rename(columns={col: f"{col}_{type}" for col in ["pos", "neu", "neg"]}, inplace=True)
        return df

    def _count_stop_words(self, tokens):
        stop_word_count = sum(1 for word in tokens if word in self.stop_words)
        return stop_word_count / len(tokens)
    
    def fitPreproccess(self, X_train):
        X_train['tokens'] = X_train['cleaned_text'].apply(lambda x: x.split())
        self.vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=True)
        tfidf_matrix = self.vectorizer.fit_transform(X_train['cleaned_text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out(), index=X_train.index)

        self.pca = PCA(n_components=6)
        principal_components = self.pca.fit_transform(tfidf_df)

        X_vec = X_train.copy()
        X_vec['len'] = X_vec['tokens'].apply(lambda x: len(x))

        X_vec = self._sentiment_make(X_vec, "cleaned_text")
        X_vec = self._sentiment_make(X_vec, "clean_title")

        X_vec['percent_stop'] = X_vec['tokens'].apply(self._count_stop_words)
        X_vec_numeric = X_vec.select_dtypes(include=np.number).drop(columns=["id"])
        principal_components_df = pd.DataFrame(principal_components)
        temp_index = X_vec.index
        X_vec_numeric.reset_index(inplace=True)
        principal_components_df.reset_index(inplace=True)
        x_train_combined = pd.concat([X_vec_numeric, principal_components_df], axis=1)
        x_train_combined.drop(columns=['index'], inplace=True)
        x_train_combined.set_index(temp_index, inplace=True)
        x_train_combined.columns = x_train_combined.columns.astype(str)
        return x_train_combined
     
    def fitModel(self, X, y):
        self.model = LogisticRegression(random_state=97)
        self.model.fit(X, y)

    def preprocessTest(self, X):
        test_tfidf_matrix = self.vectorizer.transform(X_test['cleaned_text'])

        test_tfidf = pd.DataFrame(test_tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out(), index=X_test.index)
        test_pca = pd.DataFrame(self.pca.transform(test_tfidf))


        X_test_vec = X_test.copy()
        X_test_vec['tokens'] = X_test_vec['cleaned_text'].apply(lambda x: x.split())
        X_test_vec['len'] = X_test_vec['tokens'].apply(lambda x: len(x))
        X_test_vec = self._sentiment_make(X_test_vec, "cleaned_text")
        X_test_vec = self._sentiment_make(X_test_vec, "clean_title")
        X_test_vec['percent_stop'] = X_test_vec['tokens'].apply(self._count_stop_words)

        X_vec_numeric = X_test_vec.select_dtypes(include=np.number).drop(columns=["id"])
        x_test_index = X_test.index
        X_vec_numeric.reset_index(inplace=True)
        test_pca.reset_index(inplace=True)
        x_test_processed = pd.concat([X_vec_numeric, test_pca], axis=1)
        x_test_processed.set_index(x_test_index, inplace=True)
        x_test_processed.columns = x_test_processed.columns.astype(str)
        x_test_processed.drop(columns=['index'], inplace=True)
        return x_test_processed

    def predict(self, X, cuttoff = 0.75):
        y_pred_proba = self.model.predict_proba(X)
        y_pred = [1 if proba[1] > cuttoff else 0 for proba in y_pred_proba]
        return y_pred
     


if __name__ == "__main__":
    input_file = "./article_dataset.csv"
    df_raw = pd.read_csv(input_file, index_col="id")
    df_raw = preprocessing.clean_df(df_raw)
    df_raw = df_raw[df_raw['cleaned_text'].str.strip() != ""].copy()
    df_raw.reset_index(inplace=True)
    df_raw['id'] = df_raw.index
    df_raw["clean_title"] = df_raw['title'].apply(preprocessing.clean_text)

    df = df_raw.copy()
    df = preprocessing.turn_into_pu(df, 0.5)

    X_train, X_test, y_train, y_test = preprocessing.test_train_split(df)
    y_train.index = X_train.index
    y_test_true = df_raw.loc[y_test.index, 'label']

    ### Fitting Logistic Regression ###
    print("Training Logistic Regression")
    logistic = logisticRegressionCustom()
    logistic_x_train = logistic.fitPreproccess(X_train)
    logistic.fitModel(logistic_x_train, y_train)
    logistic_x_test = logistic.preprocessTest(X_test)
    logistic_pred = logistic.predict(logistic_x_test)
    confusion_matrix = metrics.confusion_matrix(y_test_true, logistic_pred)
    print(confusion_matrix)


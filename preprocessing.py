import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
np.random.seed(5806)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


'''
Blurs true reliable news articles by making a specified percentage of reliable articles to 1
Input: Dataframe to blur on column label, Percentage of reliable sources to keep as true postives
Returns: Dataframe with blurred label column
'''
def turn_into_pu(df: pd.DataFrame, percentage:float) -> pd.DataFrame:
    zero_indices = df[df['label'] == 0].index
    stay_zero_count = int(len(zero_indices) * percentage)
    stay_zero_indices = np.random.choice(zero_indices, size=stay_zero_count, replace=False)
    df.loc[~df.index.isin(stay_zero_indices), 'label'] = 1
    return df

'''
Turns the dataset into a test train split of 30/70 stratifying on the labels
Input: Dataframe to split
Returns: X train, X test, y train, y test
'''
def test_train_split(df: pd.DataFrame):
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=5806, stratify=y)
    return X_train, X_test, y_train, y_test


'''
Basic preprocessing for text. Lowercases, removes punctuation, links, and numbers, then lemmatizes to retain only semantics
Input: string from text column
Returns: Cleaned version of that string
'''
def clean_text(text: string):
    try:
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.strip()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

        return " ".join(lemmatized)
    except:
        return ""

'''
Applies clean_text function and adds that cleaned input as a column in df
Input: dataframe
Returns: modified dataframe
'''
def clean_df(df: pd.DataFrame):
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df

if __name__=="__main__":
    input_file = "./article_dataset.csv"
    df = pd.read_csv(input_file, index_col="id")
    df = clean_df(df)
    # Remove rows with empty strings in the 'cleaned_text' column
    df = df[df['cleaned_text'].str.strip() != ""].copy()
    # with open("cleaned_dataframe.pkl", "wb") as f:
    #     pickle.dump(df, f)


    turn_into_pu(df, 0.3)
    X_train, X_test, y_train, y_test = test_train_split(df)


    print(X_train.head())
    print(df["label"].value_counts())

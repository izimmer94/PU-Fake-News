import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(5806)


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



if __name__=="__main__":
    input_file = "./article_dataset.csv"
    df = pd.read_csv(input_file, index_col="id")

    turn_into_pu(df, 0.3)
    X_train, X_test, y_train, y_test = test_train_split(df)


    print(X_train.head())
    print(df["label"].value_counts())

import sys
import warnings

import numpy as np
from pandas import read_csv

np.set_printoptions(threshold=sys.maxsize)
warnings.simplefilter('always')
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)


def import_data(features='train.csv',
                train=True):
    """Import dataset and remove row numbers column
    :return: dataframe
    """

    df = read_csv(features)
    if train:
        df.drop(['id'], axis=1, inplace=True)
    print(list(df.columns))
    print(df.head(100))
    return df


def clean_data(df):
    """Clean dataframe and impute missing values by mode
    :param df: input dataframe
    :return: clean dataframe
    """

    impute_median = df['bmi'].median()
    df['bmi'] = df['bmi'].fillna(impute_median)
    return df


if __name__ == '__main__':
    train_df = import_data(train=True)
    print(train_df.columns)

    print(train_df.isna().sum())
    train_df = clean_data(train_df)

    print('test')

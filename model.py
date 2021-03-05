import sys
import warnings

import numpy as np
from pandas import read_csv, get_dummies, concat

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
    df['gender'] = df['gender'].apply(lambda x: 1 if x.lower() == 'male' else 0)
    df['ever_married'] = df['ever_married'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    df['Residence_type'] = df['Residence_type'].apply(lambda x: 1 if x.lower() == 'urban' else 0)

    # print(df['work_type'].value_counts())
    print(df['smoking_status'].value_counts())

    return df


def one_hot_encode(df, colnames):
    """This function performs one-hot encoding of the columns
    :param df: input df
    :param colnames: columns to be one-hot encoded
    :return: dataframe
    """

    for col in colnames:
        # print(col)
        oh_df = get_dummies(df[col], prefix=col, drop_first=True)
        df = concat([oh_df, df], axis=1)
        df = df.drop([col], axis=1)

    # print(list(df.columns))
    # print(df.shape)
    return df


if __name__ == '__main__':
    train_df = import_data(train=True)
    print(train_df.columns)

    print(train_df.isna().sum())
    train_df = clean_data(train_df)
    train_df = one_hot_encode(train_df, colnames=['work_type', 'smoking_status'])



    print('test')

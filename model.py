import sys
import warnings

import numpy as np
from pandas import read_csv, get_dummies, concat, DataFrame, set_option
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(threshold=sys.maxsize)
warnings.simplefilter('always')
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)


def set_pandas():
    # Setting display option in pandas
    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    set_option('display.width', None)
    set_option('display.max_colwidth', -1)


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


def normalize_columns(df, colnames, scaler):
    """Performs Normalization using MinMaxScaler class in Sckikit-learn"""
    for col in colnames:
        x = df[[col]].values.astype(float)
        x_scaled = scaler.fit_transform(x)
        df[col] = DataFrame(x_scaled)
    print(f'''Normalized Columns: {colnames} using MinMaxScaler.''')
    return df


def split_dataset(df, test_size, seed):
    """This function randomly splits (using seed) train data into training set and validation set. The test size
    paramter specifies the ratio of input that must be allocated to the test set
    :param df: one-hot encoded dataset
    :param test_size: ratio of test-train data
    :param seed: random split
    :return: training and validation data
    """
    ncols = np.size(df, 1)
    X = df.iloc[:, range(0, ncols - 1)]

    Y = df.iloc[:, ncols - 1:]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    set_pandas()
    train_df = import_data(train=True)
    print(train_df.columns)

    print(train_df.isna().sum())
    train_df = clean_data(train_df)
    train_df = one_hot_encode(train_df, colnames=['work_type', 'smoking_status'])
    train_df = normalize_columns(train_df, colnames=['bmi', 'age', 'avg_glucose_level'], scaler=MinMaxScaler())
    x_train, x_test, y_train, y_test = split_dataset(train_df, test_size=0.2, seed=42)

    print('test')

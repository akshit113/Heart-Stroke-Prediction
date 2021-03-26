import pickle

import numpy as np
import streamlit as st
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler

from model import clean_data, normalize_columns, one_hot_encode


def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.slider('Age', 1, 100, step=1)
    hypertension = st.sidebar.selectbox('Hypertension', (1, 0))
    heart_disease = st.sidebar.selectbox('Heart Disease', (1, 0))
    ever_married = st.sidebar.selectbox('Ever Married', ('Yes', 'No'))
    work_type = st.sidebar.selectbox('Work Type',
                                     ('Private', 'Government Job', 'Never Worked', 'Children', 'Self-Employed'))
    residence_type = st.sidebar.selectbox('Residence Type', options=('Rural', 'Urban'))
    glucose_level = st.sidebar.slider('Glucose Level', 1.0, 500.0, step=0.1)
    bmi = st.sidebar.slider('BMI', 0.0, 100.0, step=0.1)
    smoking_status = st.sidebar.selectbox('Smoking Status',
                                          options=(['Smokes', 'Unknown', 'Never Smoked', 'Formerly Smoked']))
    data = {'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status}
    features = DataFrame(data, index=[0])
    return features


def collect_data():
    st.write("""
        # Heart Stroke Prediction ML App


        This app predicts whether a person is likely to suffer from heart stroke
        based on various health and lifestyle factors.

        The data for this project has been sourced from [Kaggle](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
        
        """)

    st.sidebar.header("Input Features")
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    # uploaded_file = r'train.csv'
    if uploaded_file is not None:
        input_df = read_csv(uploaded_file)
    else:
        input_df = user_input_features()
    # Combines user input features with entire training dataset
    # This will be useful for the encoding phase

    heart_raw = read_csv('train.csv')
    heart = heart_raw.drop(columns=['id', 'stroke'])
    df = concat([input_df, heart], axis=0)
    return df, len(input_df)


def preprocess_data(df, count):
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
    if 'stroke' in df.columns:
        df = df.drop(['stroke'], axis=1)

    df = clean_data(df)
    df = one_hot_encode(df, colnames=['work_type', 'smoking_status'])
    df = normalize_columns(df, colnames=['avg_glucose_level', 'bmi'], scaler=MinMaxScaler())
    print(sorted(list(df.columns)))
    df = df.iloc[:count, :]

    # print(list(df.columns))
    return df


def predictions(clf, proc_df):
    preds = (clf.predict(proc_df))
    return preds


if __name__ == '__main__':

    test_df, count = collect_data()
    proc_df = preprocess_data(test_df, count)
    # print(list(proc_df.columns))
    clf = pickle.load(open('classifier.pkl', 'rb'))

    pred = (predictions(clf, proc_df))
    # from collections import Counter
    #
    # dc = Counter(pred)
    # print(dc)
    # list_sizes = [i for i in pred if i == 1]

    print(type(pred))

    if isinstance(pred, np.ndarray):

        ls = list(pred)
        resp_list = []
        for i in ls:
            if i == 0:
                resp_list.append('The person is not exposed to the risk of heart stroke.')
            elif i == 1:
                resp_list.append('The person is at risk of heart stroke.')
            else:
                print(f'somethings wrong, i is {i}')

        from collections import Counter

        dc = Counter(resp_list)
        print(dc)
        print('')
    else:
        print('something is wrong in input. Contact administrator at akshit@email.arizona.edu')
    # else:
    #     if pred == 0:
    #         response = 'The person is not exposed to the risk of heart stroke.'
    #     else:
    #         response = 'The person is at risk of heart stroke.'

    response = "\n".join(resp_list)
    st.subheader('Prediction')
    st.write(response)

    print('done')

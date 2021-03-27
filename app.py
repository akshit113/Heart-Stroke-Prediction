import pickle

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pandas import read_csv, DataFrame, concat
from seaborn import heatmap, set_theme, diverging_palette
from sklearn.preprocessing import MinMaxScaler

from model import clean_data, normalize_columns, one_hot_encode, import_data


def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.slider('Age', 1, 100, step=1)
    hypertension = st.sidebar.selectbox('Hypertension', (1, 0))
    heart_disease = st.sidebar.selectbox('Heart Disease', (1, 0))
    ever_married = st.sidebar.selectbox('Ever Married', ('Yes', 'No'))
    work_type = st.sidebar.selectbox('Work Type',
                                     ('Private', 'Govt_job', 'Never_worked', 'children', 'Self-employed'))
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
    clf = pickle.load(open('models/xgboost.pkl', 'rb'))

    pred = (predictions(clf, proc_df))

    # from collections import Counter
    #
    # dc = Counter(pred)
    # print(dc)
    # list_sizes = [i for i in pred if i == 1]



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

    else:
        print('something is wrong in input. Contact administrator at akshit@email.arizona.edu')

    # response = "\n".join(resp_list)
    resp_df = DataFrame(resp_list, columns=['Prediction'])
    # temp = test_df.iloc[:count, :]
    # final_df = concat([resp_df,temp], axis=1)
    # print(final_df.head(10))
    st.subheader('Prediction')
    st.dataframe(resp_df)

    set_theme(style="white")

    train_df = import_data(train=True)

    corr = train_df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(3,3))

    # Generate a custom diverging colormap
    cmap = diverging_palette(256, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    # res = heatmap(corr, mask=mask, cmap=cmap, vmax=5, center=0,
    #               square=True,linewidths=2, annot=True, annot_kws={"fontsize":8})

    res = heatmap(corr, vmax=1, square=True,cmap="YlGnBu", linewidths=0.1, annot=True,
                annot_kws={"fontsize": 6},)

    res.set_xticklabels(res.get_xmajorticklabels(), fontsize=10)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize=10)
    st.pyplot(plt)

    print('done')

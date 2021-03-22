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

        The data for this project has been sourced from Kaggle
        """)

    st.sidebar.header("Input Features")
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = read_csv(uploaded_file)
    else:
        input_df = user_input_features()
    # Combines user input features with entire training dataset
    # This will be useful for the encoding phase

    heart_raw = read_csv('train.csv')
    heart = heart_raw.drop(columns=['id', 'stroke'])
    df = concat([input_df, heart], axis=0)
    return df


def preprocess_data(df):
    if 'id' in df.columns:
        df.drop(['id'], axis=1)
    if 'stroke' in df.columns:
        df.drop(['stroke'], axis=1)

    df = clean_data(df)
    df = normalize_columns(df, colnames=['avg_glucose_level', 'bmi'], scaler=MinMaxScaler())
    df = one_hot_encode(df, colnames=['work_type', 'smoking_status'])
    return df


if __name__ == '__main__':
    test_df = collect_data()
    proc_df = preprocess_data(test_df)


    print(proc_df.head())
    print(proc_df.shape)

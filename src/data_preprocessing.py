import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean(path):
    df = pd.read_csv(path)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    df.drop(['customerID'], axis=1, inplace=True)

    for col in df.select_dtypes(include='object'):
        df[col] = LabelEncoder().fit_transform(df[col])

    return df
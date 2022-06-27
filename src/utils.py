import pandas as pd
from sklearn import preprocessing


def anonymize_df(df: pd.DataFrame, col: str):  # , save_path: Path):
    """
    Anonymize dataset based on personal information column. Return df with `patient_id`
    """
    # TODO: save encoding for re-use
    le = preprocessing.LabelEncoder()
    df['patient_id'] = le.fit(df[col]).transform(df[col])

    # drop and re-order columns
    df.drop(col, axis=1, inplace=True)
    cols = list(df.columns)
    df = df[['patient_id'] + cols[:-1]]

    return df, le


def deanonymize_df(df: pd.DataFrame, col: str, le: preprocessing.LabelEncoder):
    """
    De-anonymize dataset given sklearn LabelEncoder. Return df with `patient`
    """
    df['patient'] = le.inverse_transform(df[col])
    return df

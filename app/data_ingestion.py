from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def encode_labels(df):
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])
    return df, le


def split_data(df, test_size=0.2):
    return train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        random_state=42,
        stratify=df["label"]
    )
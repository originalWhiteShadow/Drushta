import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


DEFAULT_TARGETS = {
    "Hired",
    "Loan_Approved",
    "Immediate_Care_Approved",
}


def _infer_target_column(df):
    for name in df.columns:
        if name in DEFAULT_TARGETS:
            return name
    return df.columns[-1]


def _read_dataset(file_buffer, filename=None):
    file_name = filename or getattr(file_buffer, "name", None)
    extension = os.path.splitext(file_name or "")[1].lower()

    if hasattr(file_buffer, "seek"):
        file_buffer.seek(0)

    if extension == ".tsv":
        return pd.read_csv(file_buffer, sep="\t")

    return pd.read_csv(file_buffer)


def read_dataset(file_buffer, filename=None):
    return _read_dataset(file_buffer, filename=filename)


def _encode_target(series, threshold=None, positive_label=None):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int), "binary", [0, 1]

    if pd.api.types.is_numeric_dtype(series):
        unique_values = pd.unique(series.dropna())
        if len(unique_values) == 1:
            raise ValueError("Target column has only one unique value.")
        if len(unique_values) == 2:
            sorted_values = sorted(unique_values)
            y = series.map({sorted_values[0]: 0, sorted_values[1]: 1}).astype(int)
            return y, "binary", sorted_values

        if threshold is not None:
            y = (series >= threshold).astype(int)
            return y, "binary", [f"<{threshold}", f">={threshold}"]

        encoder = LabelEncoder()
        labels = series.astype(str)
        y = encoder.fit_transform(labels)
        return y, "multiclass", encoder.classes_.tolist()

    labels = series.astype(str)
    unique_labels = sorted(pd.unique(labels.dropna()))
    if len(unique_labels) == 1:
        raise ValueError("Target column has only one unique value.")
    if len(unique_labels) == 2 and positive_label is not None:
        if positive_label not in unique_labels:
            raise ValueError("Positive label not found in target column.")
        y = (labels == positive_label).astype(int)
        negative = [label for label in unique_labels if label != positive_label][0]
        return y, "binary", [negative, positive_label]
    if len(unique_labels) == 2:
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)
        return y, "binary", encoder.classes_.tolist()

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return y, "multiclass", encoder.classes_.tolist()


def process_upload(
    source,
    filename=None,
    target_column=None,
    target_threshold=None,
    positive_label=None,
):
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = _read_dataset(source, filename=filename)

    target_column = target_column or _infer_target_column(df)

    y_raw = df[target_column].copy()
    y, target_type, class_labels = _encode_target(
        y_raw,
        threshold=target_threshold,
        positive_label=positive_label,
    )
    X = df.drop(columns=[target_column]).copy()

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))
        encoders[col] = encoder

    scaler = MinMaxScaler()
    if numeric_cols:
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return {
        "features": X,
        "target": y,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "target_column": target_column,
        "target_threshold": target_threshold,
        "positive_label": positive_label,
        "target_type": target_type,
        "class_labels": class_labels,
        "encoders": encoders,
        "scaler": scaler,
        "raw_df": df,
    }

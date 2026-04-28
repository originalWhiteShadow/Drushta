import os
import tempfile

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from backend_engines.data_engine import process_upload, read_dataset
from backend_engines.audit_engine import calculate_metrics, generate_shap_plot
from backend_engines.refine_engine import (
    apply_equalized_odds,
    apply_equalized_odds_multiclass,
    optimize_model,
)

# Make TensorFlow optional so the app can run in environments where TF
# wheels are unavailable (e.g., Streamlit Cloud Python versions).
TF_AVAILABLE = True
try:
    import tensorflow as tf
except Exception:
    tf = None
    TF_AVAILABLE = False

# Premium CSS for better aesthetics
CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.1);
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8;
    }
    div.stButton > button {
        background-color: #38bdf8;
        color: #0f172a;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #7dd3fc;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.3);
    }
    .metric-card {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.02);
    }
</style>
"""


EXAMPLE_DATASETS = {
    "HR Hiring": {
        "path": "data/hr_hiring_data.csv",
        "target": "Hired",
        "description": "Hiring outcomes with applicant attributes.",
    },
    "Finance Loans": {
        "path": "data/finance_loan_data.csv",
        "target": "Loan_Approved",
        "description": "Loan approvals with applicant attributes.",
    },
    "Medical Triage": {
        "path": "data/medical_triage_data.csv",
        "target": "Immediate_Care_Approved",
        "description": "Triage decisions with patient attributes.",
    },
}


def init_session_state():
    defaults = {
        "mode": "Upload dataset mode",
        "model_source": "Train baseline",
        "dataset_source": "Upload dataset",
        "raw_df": None,
        "raw_data_key": None,
        "dataset": None,
        "processed": None,
        "raw_model": None,
        "mitigated_model": None,
        "audit_results": None,
        "mitigated_results": None,
        "optimized_model": None,
        "thresholds": None,
        "data_key": None,
        "model_key": None,
        "model_data": None,
        "sensitive_column": None,
        "target_column": None,
        "target_threshold": None,
        "positive_label": None,
        "prediction_class": None,
        "audit_ready": False,
        "bias_scan": None,
        "bias_scan_key": None,
        "suggested_sensitive_column": None,
        "example_idx": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_model_state():
    st.session_state.raw_model = None
    st.session_state.mitigated_model = None
    st.session_state.audit_results = None
    st.session_state.mitigated_results = None
    st.session_state.optimized_model = None
    st.session_state.thresholds = None
    st.session_state.model_data = None
    st.session_state.model_key = None
    st.session_state.audit_ready = False
    st.session_state.bias_scan = None
    st.session_state.bias_scan_key = None
    st.session_state.suggested_sensitive_column = None
    st.session_state.example_idx = None


def load_dataset_from_source(raw_df, source_key, target_column, target_threshold, positive_label):
    data_key = (
        f"{source_key}:"
        f"target:{target_column}:thr:{target_threshold}:pos:{positive_label}"
    )
    processed = process_upload(
        raw_df,
        target_column=target_column,
        target_threshold=target_threshold,
        positive_label=positive_label,
    )

    st.session_state.dataset = processed["raw_df"]
    st.session_state.processed = processed
    st.session_state.data_key = data_key

    reset_model_state()


def split_data(processed):
    X = processed["features"]
    y = processed["target"].astype(int)
    indices = np.arange(len(X))

    n_samples = len(X)
    if n_samples < 2:
        raise ValueError("Dataset must have at least 2 rows for training.")

    class_counts = pd.Series(y).value_counts(dropna=False)
    test_size = 0.2 if n_samples >= 5 else 0.5
    test_count = int(np.ceil(n_samples * test_size))
    can_stratify = class_counts.min() >= 2 and len(class_counts) <= test_count

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=test_size,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    if not can_stratify:
        st.warning(
            "Some classes have fewer than 2 samples. Using a non-stratified split."
        )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "idx_train": idx_train,
        "idx_test": idx_test,
    }


def train_baseline(processed):
    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is not available in this environment. Install TensorFlow to train Keras models, or run the app locally where TensorFlow is installed."
        )
    X = processed["features"]
    model_data = split_data(processed)
    X_train = model_data["X_train"]
    y_train = model_data["y_train"]

    target_type = processed.get("target_type", "binary")
    class_labels = processed.get("class_labels") or [0, 1]
    num_classes = len(class_labels)

    if target_type == "multiclass" and num_classes > 2:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(X.shape[1],)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(X.shape[1],)),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    model.fit(X_train, y_train, epochs=3, batch_size=16, verbose=0)

    return model, model_data


def load_uploaded_model(uploaded_model, processed):
    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is not available in this environment. Uploaded Keras models cannot be loaded without TensorFlow."
        )
    model_data = split_data(processed)
    suffix = os.path.splitext(uploaded_model.name)[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_model.getvalue())
        temp_path = temp_file.name

    try:
        model = tf.keras.models.load_model(temp_path, compile=False)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    expected_features = processed["features"].shape[1]
    model_features = model.input_shape[-1] if model.input_shape else None

    target_type = processed.get("target_type", "binary")
    class_labels = processed.get("class_labels") or [0, 1]
    num_classes = len(class_labels)
    output_shape = model.output_shape[-1] if model.output_shape else None

    if model_features is not None and model_features != expected_features:
        raise ValueError(
            f"Model expects {model_features} features but dataset has {expected_features}."
        )

    if target_type == "multiclass" and num_classes > 2:
        if output_shape is not None and output_shape != num_classes:
            raise ValueError(
                f"Model outputs {output_shape} classes but dataset has {num_classes}."
            )
    else:
        if output_shape is not None and output_shape != 1:
            raise ValueError("Binary targets require a single-output (sigmoid) model.")

    return model, model_data


def get_sensitive_default(df, target_column):
    candidates = [col for col in df.columns if col != target_column]
    if not candidates:
        return None

    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in candidates:
        if col in categorical:
            return col

    return candidates[0]


def build_input_form(processed):
    raw_df = processed["raw_df"]
    numeric_cols = processed["numeric_cols"]
    categorical_cols = processed["categorical_cols"]

    user_values = {}
    for col in processed["features"].columns:
        if col in categorical_cols:
            options = raw_df[col].dropna().astype(str).unique().tolist()
            user_values[col] = st.selectbox(col, options)
        else:
            col_min = float(raw_df[col].min())
            col_max = float(raw_df[col].max())
            if pd.api.types.is_integer_dtype(raw_df[col]):
                user_values[col] = st.slider(col, int(col_min), int(col_max), int(col_min))
            else:
                user_values[col] = st.number_input(col, min_value=col_min, max_value=col_max, value=col_min)

    return user_values


def transform_input(user_values, processed):
    features = processed["features"]
    encoders = processed["encoders"]
    scaler = processed["scaler"]

    input_df = pd.DataFrame([user_values])

    for col, encoder in encoders.items():
        try:
            input_df[col] = encoder.transform(input_df[col].astype(str))
        except ValueError:
            input_df[col] = 0

    numeric_cols = processed["numeric_cols"]
    if numeric_cols:
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    return input_df[features.columns]


def compute_audit_results(processed, model_data, sensitive_column):
    raw_df = processed["raw_df"]
    X_test = model_data["X_test"]
    y_test = model_data["y_test"]
    idx_test = model_data["idx_test"]

    target_type = processed.get("target_type", "binary")
    probabilities = st.session_state.raw_model.predict(X_test, verbose=0)

    if target_type == "multiclass" and probabilities.ndim > 1:
        y_pred = np.argmax(probabilities, axis=1)
    else:
        probabilities = probabilities.reshape(-1)
        y_pred = (probabilities >= 0.5).astype(int)

    sensitive_values = raw_df.iloc[idx_test][sensitive_column].values
    results = calculate_metrics(y_test, y_pred, sensitive_values)

    return results, probabilities, sensitive_values


def _metric_spread(results):
    if not results:
        return 0.0, 0.0

    fnr_values = [metrics["FNR"] for metrics in results.values()]
    fpr_values = [metrics["FPR"] for metrics in results.values()]

    return (max(fnr_values) - min(fnr_values)), (max(fpr_values) - min(fpr_values))


def scan_sensitive_columns(processed, model_data, max_unique=12):
    raw_df = processed["raw_df"]
    target_column = processed["target_column"]
    candidate_cols = [col for col in raw_df.columns if col != target_column]

    X_test = model_data["X_test"]
    y_test = model_data["y_test"]
    idx_test = model_data["idx_test"]

    target_type = processed.get("target_type", "binary")
    probabilities = st.session_state.raw_model.predict(X_test, verbose=0)
    if target_type == "multiclass" and probabilities.ndim > 1:
        y_pred = np.argmax(probabilities, axis=1)
    else:
        probabilities = probabilities.reshape(-1)
        y_pred = (probabilities >= 0.5).astype(int)

    results = []
    for col in candidate_cols:
        series = raw_df[col]
        unique_count = series.nunique(dropna=True)
        if unique_count < 2 or unique_count > max_unique:
            continue

        sensitive_values = raw_df.iloc[idx_test][col].values
        metrics = calculate_metrics(y_test, y_pred, sensitive_values)
        fnr_spread, fpr_spread = _metric_spread(metrics)
        score = max(fnr_spread, fpr_spread)

        results.append(
            {
                "column": col,
                "fnr_spread": fnr_spread,
                "fpr_spread": fpr_spread,
                "score": score,
                "groups": unique_count,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


st.set_page_config(
    page_title="Drushta AI | Universal Bias Auditor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
init_session_state()

st.title("⚖️ Drushta AI - Universal Bias Auditor")
st.markdown("---")

with st.sidebar:
    st.header("Data & Model")
    available_examples = {
        name: info
        for name, info in EXAMPLE_DATASETS.items()
        if os.path.exists(info["path"])
    }
    dataset_options = ["Upload dataset"]
    if available_examples:
        dataset_options.append("Example dataset")

    dataset_source = st.radio(
        "Dataset Source",
        dataset_options,
        horizontal=True,
    )
    if dataset_source == "Example dataset" and not available_examples:
        dataset_source = "Upload dataset"
        st.info("Example datasets are not available. Add files under data/ to enable them.")

    if dataset_source != st.session_state.dataset_source:
        st.session_state.dataset_source = dataset_source
        reset_model_state()

    mode = st.radio(
        "Mode",
        ["Upload dataset mode", "Upload model mode"],
        help="Dataset mode trains a baseline model. Model mode audits an uploaded model.",
    )
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        reset_model_state()

    uploaded_file = None
    example_name = None
    example_path = None

    if dataset_source == "Upload dataset":
        uploaded_file = st.file_uploader("Upload dataset (CSV or TSV)", type=["csv", "tsv"])
        st.caption("Target column defaults to the last column, or a known label like Hired/Loan_Approved.")
    else:
        example_name = st.selectbox("Example dataset", list(available_examples.keys()))
        example_info = available_examples[example_name]
        example_path = example_info["path"]
        st.caption(example_info["description"])
        if example_path:
            try:
                with open(example_path, "rb") as dataset_file:
                    dataset_bytes = dataset_file.read()
                st.download_button(
                    "Download dataset",
                    data=dataset_bytes,
                    file_name=os.path.basename(example_path),
                    mime="text/csv",
                )
            except OSError as exc:
                st.error(f"Failed to load example dataset for download: {exc}")

    raw_df = None
    if dataset_source == "Upload dataset":
        if uploaded_file is None:
            if st.session_state.raw_data_key is not None:
                st.session_state.raw_df = None
                st.session_state.raw_data_key = None
                st.session_state.target_column = None
                st.session_state.target_threshold = None
                st.session_state.positive_label = None
                st.session_state.dataset = None
                st.session_state.processed = None
                st.session_state.data_key = None
                st.session_state.audit_ready = False
                reset_model_state()
        else:
            raw_key = f"upload:{uploaded_file.name}:{uploaded_file.size}"
            if st.session_state.raw_data_key != raw_key:
                try:
                    raw_df = read_dataset(uploaded_file, filename=uploaded_file.name)
                    st.session_state.raw_df = raw_df
                    st.session_state.raw_data_key = raw_key
                    st.session_state.target_column = None
                    st.session_state.target_threshold = None
                    st.session_state.positive_label = None
                except Exception as exc:
                    st.session_state.raw_df = None
                    st.session_state.raw_data_key = None
                    st.error(f"Failed to read dataset: {exc}")
            raw_df = st.session_state.raw_df
    else:
        raw_key = f"example:{example_name}:{example_path}"
        if st.session_state.raw_data_key != raw_key:
            try:
                raw_df = read_dataset(example_path, filename=example_path)
                st.session_state.raw_df = raw_df
                st.session_state.raw_data_key = raw_key
                st.session_state.target_column = available_examples[example_name].get("target")
                st.session_state.target_threshold = None
                st.session_state.positive_label = None
            except Exception as exc:
                st.session_state.raw_df = None
                st.session_state.raw_data_key = None
                st.error(f"Failed to read dataset: {exc}")
        raw_df = st.session_state.raw_df

    if raw_df is not None:
        columns = list(raw_df.columns)
        if not columns:
            st.error("Dataset has no columns.")
        else:
            default_target = st.session_state.target_column or columns[-1]
            st.caption(f"Rows: {len(raw_df)} | Columns: {len(columns)}")
            target_column = st.selectbox(
                "Target column",
                columns,
                index=columns.index(default_target) if default_target in columns else 0,
            )
            st.session_state.target_column = target_column

            target_series = raw_df[target_column]
            target_threshold = None
            positive_label = None
            target_error = None
            target_info = None

            if pd.api.types.is_numeric_dtype(target_series):
                unique_values = pd.unique(target_series.dropna())
                if len(unique_values) > 2:
                    col_min = float(np.nanmin(target_series))
                    col_max = float(np.nanmax(target_series))
                    if col_min == col_max:
                        target_error = "Target column has only one unique value."
                    else:
                        use_threshold = st.checkbox(
                            "Binarize numeric target",
                            help="If unchecked, the target is treated as multiclass.",
                        )
                        if use_threshold:
                            default_threshold = float(np.nanmedian(target_series))
                            target_threshold = st.slider(
                                "Target threshold",
                                min_value=col_min,
                                max_value=col_max,
                                value=default_threshold,
                            )
                        else:
                            target_info = f"Using multiclass target with {len(unique_values)} classes."
                elif len(unique_values) < 2:
                    target_error = "Target column has only one unique value."
            else:
                unique_labels = sorted(pd.unique(target_series.dropna().astype(str)))
                if len(unique_labels) == 2:
                    positive_label = st.selectbox(
                        "Positive class label",
                        unique_labels,
                        index=1,
                    )
                else:
                    target_info = f"Using multiclass target with {len(unique_labels)} classes."

            st.session_state.target_threshold = target_threshold
            st.session_state.positive_label = positive_label

            if target_error:
                st.error(target_error)
                st.session_state.processed = None
                st.session_state.dataset = None
                st.session_state.data_key = None
                reset_model_state()
            else:
                if target_info:
                    st.info(target_info)
                source_key = st.session_state.raw_data_key
                data_key = (
                    f"{source_key}:"
                    f"target:{target_column}:thr:{target_threshold}:pos:{positive_label}"
                )
                class_counts = target_series.value_counts(dropna=False).head(6)
                st.caption(
                    "Top classes: "
                    + ", ".join([f"{idx}={val}" for idx, val in class_counts.items()])
                )
                if st.session_state.data_key != data_key:
                    try:
                        load_dataset_from_source(
                            raw_df,
                            source_key,
                            target_column,
                            target_threshold,
                            positive_label,
                        )
                    except ValueError as exc:
                        st.session_state.processed = None
                        st.session_state.dataset = None
                        st.session_state.data_key = None
                        reset_model_state()
                        st.error(str(exc))

    model_source = "Upload model" if mode == "Upload model mode" else "Train baseline"
    uploaded_model = None
    if model_source == "Upload model":
        uploaded_model = st.file_uploader(
            "Upload Keras Model",
            type=["keras", "h5"],
            help="Model input features must match the uploaded dataset feature order.",
        )

    if st.session_state.processed is not None:
        target_col = st.session_state.processed["target_column"]
        default_sensitive = get_sensitive_default(st.session_state.dataset, target_col)
        candidate_cols = [col for col in st.session_state.dataset.columns if col != target_col]

        if not candidate_cols:
            st.warning("No sensitive columns available.")
            st.session_state.sensitive_column = None
        else:
            if (
                st.session_state.sensitive_column is None
                or st.session_state.sensitive_column not in st.session_state.dataset.columns
            ):
                st.session_state.sensitive_column = default_sensitive

            previous_sensitive = st.session_state.sensitive_column
            st.session_state.sensitive_column = st.selectbox(
                "Sensitive Column",
                candidate_cols,
                index=candidate_cols.index(st.session_state.sensitive_column),
            )
            if st.session_state.sensitive_column != previous_sensitive:
                st.session_state.audit_results = None
                st.session_state.mitigated_results = None
                st.session_state.thresholds = None

        if st.session_state.processed.get("target_type") == "multiclass":
            class_labels = st.session_state.processed.get("class_labels") or []
            if class_labels:
                if st.session_state.prediction_class not in class_labels:
                    st.session_state.prediction_class = class_labels[0]

                st.session_state.prediction_class = st.selectbox(
                    "Output class to predict",
                    class_labels,
                    index=class_labels.index(st.session_state.prediction_class),
                )

    if model_source != st.session_state.model_source:
        st.session_state.model_source = model_source
        reset_model_state()

if st.session_state.processed is not None:
    if st.session_state.model_source == "Upload model":
        if uploaded_model is None:
            st.warning("Upload a Keras model to proceed.")
        else:
            model_key = f"upload_model:{uploaded_model.name}:{uploaded_model.size}"
            if st.session_state.raw_model is None or st.session_state.model_key != model_key:
                try:
                    model, model_data = load_uploaded_model(uploaded_model, st.session_state.processed)
                    st.session_state.raw_model = model
                    st.session_state.model_data = model_data
                    st.session_state.model_key = model_key
                except ValueError as exc:
                    st.session_state.raw_model = None
                    st.session_state.model_data = None
                    st.error(str(exc))
    else:
        if st.session_state.raw_model is None:
            model, model_data = train_baseline(st.session_state.processed)
            st.session_state.raw_model = model
            st.session_state.model_data = model_data

    if (
        st.session_state.mode == "Upload dataset mode"
        and st.session_state.raw_model is not None
        and st.session_state.model_data is not None
    ):
        scan_key = f"{st.session_state.data_key}:{st.session_state.model_key or 'baseline'}"
        if st.session_state.bias_scan_key != scan_key:
            st.session_state.bias_scan = scan_sensitive_columns(
                st.session_state.processed, st.session_state.model_data
            )
            st.session_state.bias_scan_key = scan_key
            if st.session_state.bias_scan:
                st.session_state.suggested_sensitive_column = st.session_state.bias_scan[0][
                    "column"
                ]

        if st.session_state.bias_scan:
            top_hit = st.session_state.bias_scan[0]
            suggestion = "Apply Equalized Odds" if top_hit["score"] >= 0.1 else "Review data or proceed without mitigation"
            st.info(
                "Bias scan suggestion: use the sensitive column "
                f"'{top_hit['column']}' (FNR spread {top_hit['fnr_spread']:.2f}, "
                f"FPR spread {top_hit['fpr_spread']:.2f}). Recommended fix: {suggestion}."
            )
            if st.button("Proceed to audit", type="primary"):
                st.session_state.audit_ready = True
        else:
            st.info("Bias scan needs a categorical or low-cardinality column to analyze.")
            if st.button("Proceed without bias scan"):
                st.session_state.audit_ready = True
            else:
                st.session_state.audit_ready = False

    if st.session_state.mode == "Upload model mode" and st.session_state.raw_model is not None:
        st.session_state.audit_ready = True

    if (
        st.session_state.raw_model is not None
        and st.session_state.model_data is not None
        and st.session_state.audit_results is None
        and st.session_state.sensitive_column is not None
        and st.session_state.audit_ready
    ):
        results, probabilities, sensitive_values = compute_audit_results(
            st.session_state.processed, st.session_state.model_data, st.session_state.sensitive_column
        )
        st.session_state.audit_results = results
        st.session_state.model_data["probabilities"] = probabilities
        st.session_state.model_data["sensitive_values"] = sensitive_values


# Tabs

tab_audit, tab_mitigate, tab_live, tab_export = st.tabs(
    ["📥 Audit", "⚙️ Mitigate", "🔍 Live Validation", "📦 Export"]
)

with tab_audit:
    st.subheader("Baseline Fairness Metrics")
    if not st.session_state.audit_ready:
        st.info("Review the bias scan and click 'Proceed to audit' to continue.")
    elif st.session_state.audit_results is None:
        st.info("Awaiting audit results.")
    else:
        results = st.session_state.audit_results
        
        # Display summary metrics in columns
        col1, col2, col3 = st.columns(3)
        table = pd.DataFrame.from_dict(results, orient="index")
        
        avg_fnr = table["FNR"].mean()
        avg_fpr = table["FPR"].mean()
        max_disparity = table["FNR"].max() - table["FNR"].min()
        
        col1.metric("Avg FNR", f"{avg_fnr:.3f}")
        col2.metric("Avg FPR", f"{avg_fpr:.3f}")
        col3.metric("Max Disparity (FNR)", f"{max_disparity:.3f}", delta=f"{-max_disparity:.3f}", delta_color="inverse")

        st.markdown("### Metrics by Group")
        st.dataframe(table.style.background_gradient(cmap="RdYlGn_r", subset=["FNR", "FPR"]), use_container_width=True)

        st.markdown("### Disparity Analysis")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.caption("False Negative Rate Disparity")
            st.bar_chart(table["FNR"])
        with col_c2:
            st.caption("False Positive Rate Disparity")
            st.bar_chart(table["FPR"])

with tab_mitigate:
    st.subheader("Mitigation Controls")
    if not st.session_state.audit_ready:
        st.info("Confirm the bias scan to enable mitigation controls.")
    else:
        target_type = (
            st.session_state.processed.get("target_type", "binary")
            if st.session_state.processed
            else "binary"
        )
        if target_type == "multiclass":
            st.info(
                "Equalized Odds uses a one-vs-rest approximation for multiclass targets."
            )

        if st.session_state.mode == "Upload model mode":
            st.info(
                "Suggested fixes: apply Equalized Odds for group disparity, then prune & quantize "
                "for a lighter model. Other options include weight clustering or distillation."
            )

        apply_eq_odds = st.checkbox(
            "Apply Equalized Odds",
            value=st.session_state.mode == "Upload model mode",
        )
        apply_prune = st.checkbox("Prune & Quantize (Edge Ready)")

    if st.session_state.audit_ready and st.button("Run Mitigation"):
        progress = st.progress(0)
        progress.progress(20)

        if apply_eq_odds:
            probabilities = st.session_state.model_data["probabilities"]
            sensitive_values = st.session_state.model_data["sensitive_values"]
            y_test = st.session_state.model_data["y_test"]
            if probabilities is None:
                st.error("Equalized Odds requires prediction probabilities.")
            else:
                if target_type == "multiclass":
                    class_labels = st.session_state.processed.get("class_labels") or []
                    mitigated_pred, thresholds = apply_equalized_odds_multiclass(
                        probabilities,
                        y_test,
                        sensitive_values,
                        class_labels,
                    )
                else:
                    mitigated_pred, thresholds = apply_equalized_odds(
                        probabilities, sensitive_values
                    )

                st.session_state.thresholds = thresholds
                st.session_state.mitigated_results = calculate_metrics(
                    y_test, mitigated_pred, sensitive_values
                )
                st.session_state.mitigated_model = st.session_state.raw_model

        progress.progress(70)

        if apply_prune:
            st.session_state.optimized_model = optimize_model(st.session_state.raw_model)

        progress.progress(100)
        st.success("Mitigation complete.")

    if st.session_state.mitigated_results is not None:
        st.markdown("---")
        st.markdown("### Mitigated Fairness Comparison")
        
        m_results = st.session_state.mitigated_results
        m_table = pd.DataFrame.from_dict(m_results, orient="index")
        
        col_b1, col_a1 = st.columns(2)
        
        with col_b1:
            st.markdown("#### Baseline (Original)")
            st.dataframe(pd.DataFrame.from_dict(st.session_state.audit_results, orient="index").style.background_gradient(cmap="RdYlGn_r", subset=["FNR", "FPR"]), use_container_width=True)
            
        with col_a1:
            st.markdown("#### Mitigated")
            st.dataframe(m_table.style.background_gradient(cmap="RdYlGn_r", subset=["FNR", "FPR"]), use_container_width=True)

        # Delta visualization
        st.markdown("#### Disparity Improvement")
        baseline_table = pd.DataFrame.from_dict(st.session_state.audit_results, orient="index")
        delta_fnr = (m_table["FNR"].max() - m_table["FNR"].min()) - (baseline_table["FNR"].max() - baseline_table["FNR"].min())
        
        st.metric("Disparity Reduction (FNR)", f"{delta_fnr:.4f}", delta=f"{delta_fnr:.4f}", delta_color="normal")

with tab_live:
    st.subheader("Live Validation")
    if st.session_state.processed is None or st.session_state.raw_model is None:
        st.info("Load a dataset to begin.")
    elif st.session_state.model_data is None:
        st.info("Awaiting model data.")
    else:
        if st.button("Pick new example") or st.session_state.example_idx is None:
            st.session_state.example_idx = int(
                np.random.choice(st.session_state.model_data["idx_test"])
            )

        example_idx = st.session_state.example_idx
        raw_example = st.session_state.processed["raw_df"].iloc[example_idx]
        user_values = raw_example.to_dict()

        st.markdown("### Real-time Feature Toggling")
        
        with st.expander("📝 Edit Features Manually", expanded=True):
            col_in1, col_in2 = st.columns(2)
            
            # Split features into two columns for better layout
            feature_cols = list(st.session_state.processed["features"].columns)
            mid = len(feature_cols) // 2
            
            user_values = {}
            raw_df = st.session_state.processed["raw_df"]
            categorical_cols = st.session_state.processed["categorical_cols"]
            
            # Helper to build input for a feature
            def feature_input(col):
                if col in categorical_cols:
                    options = sorted(raw_df[col].dropna().astype(str).unique().tolist())
                    default_val = str(raw_example[col])
                    return st.selectbox(col, options, index=options.index(default_val) if default_val in options else 0)
                else:
                    col_min = float(raw_df[col].min())
                    col_max = float(raw_df[col].max())
                    curr_val = float(raw_example[col])
                    if pd.api.types.is_integer_dtype(raw_df[col]):
                        return st.slider(col, int(col_min), int(col_max), int(curr_val))
                    else:
                        return st.number_input(col, min_value=col_min, max_value=col_max, value=curr_val)

            with col_in1:
                for col in feature_cols[:mid]:
                    user_values[col] = feature_input(col)
            with col_in2:
                for col in feature_cols[mid:]:
                    user_values[col] = feature_input(col)

        processed_input = transform_input(user_values, st.session_state.processed)
        target_type = st.session_state.processed.get("target_type", "binary")
        class_labels = st.session_state.processed.get("class_labels") or [0, 1]

        prediction = st.session_state.raw_model.predict(processed_input, verbose=0)
        threshold = 0.5

        if target_type == "multiclass" and prediction.ndim > 1:
            probs = prediction[0]
            raw_pred_idx = int(np.argmax(probs))
            raw_pred_label = class_labels[raw_pred_idx]
            selected_class = st.session_state.prediction_class or raw_pred_label
            if selected_class not in class_labels:
                selected_class = raw_pred_label

            selected_idx = class_labels.index(selected_class)
            raw_prob = float(probs[selected_idx])
            mitigated_pred_label = raw_pred_label
            selected_threshold = 1.0 / max(len(class_labels), 1)

            if st.session_state.thresholds and st.session_state.sensitive_column in user_values:
                group_value = str(user_values[st.session_state.sensitive_column])
                group_thresholds = st.session_state.thresholds.get(group_value, {})
                adjusted_scores = []
                for class_idx, label in enumerate(class_labels):
                    threshold = float(group_thresholds.get(label, 1.0 / max(len(class_labels), 1)))
                    adjusted_scores.append(probs[class_idx] - threshold)

                mitigated_pred_idx = int(np.argmax(adjusted_scores))
                mitigated_pred_label = class_labels[mitigated_pred_idx]
                selected_threshold = float(
                    group_thresholds.get(selected_class, selected_threshold)
                )
        else:
            raw_prob = float(prediction.reshape(-1)[0])
            if st.session_state.thresholds and st.session_state.sensitive_column in user_values:
                group_value = str(user_values[st.session_state.sensitive_column])
                threshold = float(st.session_state.thresholds.get(group_value, 0.5))

            raw_pred_label = class_labels[1] if raw_prob >= 0.5 else class_labels[0]
            mitigated_pred_label = class_labels[1] if raw_prob >= threshold else class_labels[0]

        st.markdown("---")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if target_type == "multiclass":
                st.metric("Before Mitigation", f"{selected_class}", delta=f"p={raw_prob:.2f}", delta_color="off")
                st.caption(f"Top class: {raw_pred_label}")
            else:
                st.metric("Before Mitigation", f"{raw_pred_label}", delta=f"p={raw_prob:.2f}", delta_color="off")
            
            with st.spinner("Generating SHAP..."):
                fig_raw = generate_shap_plot(st.session_state.raw_model, processed_input)
            if fig_raw is None:
                st.info("SHAP plot unavailable for this input.")
            else:
                st.pyplot(fig_raw, clear_figure=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if target_type == "multiclass":
                st.metric(
                    "After Mitigation",
                    f"{selected_class}",
                    delta=f"t={selected_threshold:.2f}",
                    delta_color="normal"
                )
                st.caption(f"Top class after: {mitigated_pred_label}")
            else:
                st.metric("After Mitigation", f"{mitigated_pred_label}", delta=f"t={threshold:.2f}", delta_color="normal")
            
            # Post-processing note
            st.info("💡 Post-processing mitigation adjusts thresholds, not model weights. Feature importance (SHAP) remains consistent.")
            
            with st.spinner("Generating SHAP..."):
                fig_mitigated = generate_shap_plot(st.session_state.raw_model, processed_input)
            if fig_mitigated is None:
                st.info("SHAP plot unavailable for this input.")
            else:
                st.pyplot(fig_mitigated, clear_figure=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Prediction Output Graph")
        if target_type == "multiclass":
            chart_data = pd.DataFrame(
                {
                    "Probability": [raw_prob, raw_prob],
                },
                index=["Before", "After"],
            )
        else:
            chart_data = pd.DataFrame(
                {
                    "Probability": [raw_prob, raw_prob],
                    "Decision Threshold": [0.5, threshold],
                },
                index=["Before", "After"],
            )
        st.bar_chart(chart_data, use_container_width=True)

with tab_export:
    st.subheader("📦 Model Deployment Hub")
    
    col_e1, col_e2 = st.columns([2, 1])
    
    with col_e1:
        st.markdown("""
        ### Audit Certification
        Your model has been audited for bias and optimized for edge deployment. 
        Below is the deployment readiness summary.
        """)
        
        score = 40
        checks = []
        if st.session_state.audit_results is not None:
            score += 20
            checks.append("✅ Fairness Audit Completed")
        else:
            checks.append("❌ Fairness Audit Pending")
            
        if st.session_state.thresholds is not None:
            score += 20
            checks.append("✅ Bias Mitigation Applied")
        else:
            checks.append("❌ Bias Mitigation Pending")
            
        if st.session_state.optimized_model is not None:
            score += 20
            checks.append("✅ Edge Optimization (Pruning/Quantization) Completed")
        else:
            checks.append("❌ Edge Optimization Pending")

        for check in checks:
            st.write(check)

    with col_e2:
        score = min(score, 100)
        st.metric("Readiness Score", f"{score}/100")
        
        if score >= 80:
            st.success("High Readiness: Model is ready for production.")
        elif score >= 60:
            st.warning("Medium Readiness: Consider additional optimization.")
        else:
            st.error("Low Readiness: Audit and mitigate before deployment.")

    st.markdown("---")
    if st.session_state.optimized_model is None:
        st.info("💡 To enable export, run mitigation with 'Prune & Quantize' selected in the Mitigate tab.")
    else:
        st.download_button(
            "🚀 Download Edge-Optimized Model (.tflite)",
            data=st.session_state.optimized_model,
            file_name="drushta_optimized_model.tflite",
            mime="application/octet-stream",
            use_container_width=True
        )

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report,
)

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "model" / "combined models.joblib"
TARGET_DEFAULT = "diagnosis"

st.set_page_config(page_title="Breast Cancer Classification | AIML Assignment 2",
                   page_icon="📊", layout="wide")

@st.cache_resource
def load_models(path):
    # Load only model artifacts you trust. Keep the sklearn version compatible
    # with the version used when the artifact was saved.
    return joblib.load(path)

st.title("📊 Breast Cancer Classification Dashboard")
st.caption("M.Tech AIML · Machine Learning Assignment 2")
st.markdown("""
Evaluate the five trained classifiers on a **held-out test CSV**. This application
loads saved estimators and performs inference only; it does not retrain models.
""")

if not MODEL_FILE.exists():
    st.error(f"Saved model file not found: `{MODEL_FILE}`. Ensure the model folder is included in the repository.")
    st.stop()

try:
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        models = load_models(str(MODEL_FILE))
except Exception as exc:
    st.error(f"Could not load the saved model bundle: {exc}")
    st.info("Check that requirements.txt uses the same scikit-learn version used to save the model.")
    st.stop()

if not isinstance(models, dict) or not models:
    st.error("Expected the model bundle to be a dictionary of model name → trained estimator.")
    st.stop()

with st.sidebar:
    st.header("Evaluation settings")
    chosen_model = st.selectbox("Model to inspect", list(models.keys()))
    st.write("**Target column:** diagnosis")
    st.caption("Upload a CSV with the 30 feature columns and the actual diagnosis labels (0 = benign, 1 = malignant).")

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
if uploaded is None:
    st.info("You can try the provided test file: `Data/BreastCancerWisconsinTestRecords.csv`.")
    if st.button("Load the included test data"):
        st.session_state["use_sample"] = True
    if not st.session_state.get("use_sample", False):
        st.stop()
    test_path = ROOT / "Data" / "BreastCancerWisconsinTestRecords.csv"
    if not test_path.exists():
        st.error("Included test data file was not found in Data/.")
        st.stop()
    data = pd.read_csv(test_path)
else:
    try:
        data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Unable to read the uploaded CSV: {exc}")
        st.stop()

if data.empty:
    st.error("The CSV contains no rows.")
    st.stop()
if TARGET_DEFAULT not in data.columns:
    st.error("CSV must contain the target column named `diagnosis`.")
    st.stop()

y = data[TARGET_DEFAULT]
if y.isna().any():
    st.error("The target column contains missing values. Please provide actual labels for every test row.")
    st.stop()

# Expected feature list from the notebook: drop id, diagnosis, and the empty Unnamed column.
# The saved estimators were fitted using these 30 input columns.
X = data.drop(columns=[TARGET_DEFAULT])
X = X.drop(columns=[c for c in ["id", "Unnamed: 32"] if c in X.columns], errors="ignore")

reference_estimator = next(iter(models.values()))
expected = getattr(reference_estimator, "feature_names_in_", None)
if expected is not None:
    expected = list(expected)
    missing = [col for col in expected if col not in X.columns]
    extra = [col for col in X.columns if col not in expected]
    if missing:
        st.error("Missing required feature columns: " + ", ".join(missing))
        st.stop()
    X = X[expected]
elif X.shape[1] != 30:
    st.error(f"Expected 30 feature columns after removing target/id/empty columns; found {X.shape[1]}.")
    st.stop()

try:
    X = X.apply(pd.to_numeric, errors="raise")
except Exception as exc:
    st.error(f"All feature values must be numeric. Details: {exc}")
    st.stop()

st.subheader("Test data overview")
m1, m2, m3 = st.columns(3)
m1.metric("Test records", f"{len(data):,}")
m2.metric("Input features", f"{X.shape[1]}")
m3.metric("Target classes", f"{y.nunique()}")
with st.expander("Preview uploaded test records"):
    st.dataframe(data.head(12), use_container_width=True)

def evaluate(estimator, features, actual):
    predicted = estimator.predict(features)
    unique_classes = np.unique(actual)
    try:
        if hasattr(estimator, "predict_proba"):
            score = estimator.predict_proba(features)
            if score.ndim == 2 and score.shape[1] == 2:
                auc = roc_auc_score(actual, score[:, 1])
            else:
                auc = roc_auc_score(actual, score, multi_class="ovr", average="weighted")
        elif hasattr(estimator, "decision_function"):
            score = estimator.decision_function(features)
            auc = roc_auc_score(actual, score, multi_class="ovr", average="weighted")
        else:
            auc = np.nan
    except Exception:
        auc = np.nan

    metrics = {
        "Accuracy": accuracy_score(actual, predicted),
        "AUC": auc,
        "Precision": precision_score(actual, predicted, average="binary", zero_division=0),
        "Recall": recall_score(actual, predicted, average="binary", zero_division=0),
        "F1": f1_score(actual, predicted, average="binary", zero_division=0),
        "MCC": matthews_corrcoef(actual, predicted),
    }
    return predicted, metrics

st.subheader("Model comparison")
comparison = []
predictions_by_model = {}
errors = {}
for name, estimator in models.items():
    try:
        prediction, metrics = evaluate(estimator, X, y)
        predictions_by_model[name] = prediction
        comparison.append({"ML Model Name": name, **metrics})
    except Exception as exc:
        errors[name] = str(exc)

if comparison:
    comparison_df = pd.DataFrame(comparison).set_index("ML Model Name")
    st.dataframe(comparison_df.style.format("{:.4f}", na_rep="N/A"), use_container_width=True)
    st.download_button("Download model comparison CSV",
                       comparison_df.reset_index().to_csv(index=False).encode("utf-8"),
                       file_name="model_comparison_metrics.csv", mime="text/csv")
if errors:
    with st.expander("Models with evaluation errors"):
        for name, message in errors.items():
            st.warning(f"{name}: {message}")

if chosen_model not in predictions_by_model:
    st.error(f"The selected model could not be evaluated. {errors.get(chosen_model, '')}")
    st.stop()

st.divider()
st.subheader(f"Selected model: {chosen_model}")
predicted = predictions_by_model[chosen_model]
selected_metrics = comparison_df.loc[chosen_model]
metric_cols = st.columns(6)
for col, metric_name in zip(metric_cols, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    value = selected_metrics[metric_name]
    col.metric(metric_name, "N/A" if pd.isna(value) else f"{value:.4f}")

labels = sorted(pd.unique(y))
cm = confusion_matrix(y, predicted, labels=labels)
st.subheader("Confusion matrix")
cm_df = pd.DataFrame(cm, index=[f"Actual {label}" for label in labels],
                     columns=[f"Predicted {label}" for label in labels])
st.dataframe(cm_df, use_container_width=True)

st.subheader("Classification report")
report = classification_report(y, predicted, zero_division=0, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

st.subheader("Predictions")
result = data.copy()
result[f"Predicted_{TARGET_DEFAULT}"] = predicted
st.dataframe(result, use_container_width=True)
st.download_button("Download predictions CSV", result.to_csv(index=False).encode("utf-8"),
                   file_name=f"predictions_{chosen_model.lower().replace(' ', '_')}.csv",
                   mime="text/csv")

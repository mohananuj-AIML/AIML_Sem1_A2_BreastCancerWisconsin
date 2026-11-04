README.txt
===========

PROJECT
-------
Streamlit app for Machine Learning Assignment 2 (M.Tech AIML).
Dataset: Breast Cancer Wisconsin Diagnostic Dataset.
The app loads the five saved trained estimators and evaluates uploaded test CSV data.

FILES
-----
app.py
    Streamlit UI and model inference/evaluation.
requirements.txt
    Runtime Python dependencies.
Data/
    Dataset and held-out test records CSV.
model/combined models.joblib
    Dictionary containing the saved trained estimators:
    Logistic Regression, Decision Tree, KNN, Naive Bayes, and Support Vector Machine.

FEATURES
--------
- Upload test records as CSV (target column must be diagnosis).
- Model selection dropdown.
- Side-by-side evaluation metrics for all five models.
- Metrics for selected model: Accuracy, AUC, Precision, Recall, F1, MCC.
- Confusion matrix and classification report.
- Preview and download predictions and model comparison metrics.
- Uses saved models; does not retrain.

LOCAL RUN
---------
1. Install Python. A Python version supported by the pinned scikit-learn version is recommended.
2. Open a terminal at the project root (same directory as app.py).
3. Install packages:
       python -m pip install -r requirements.txt
4. Start Streamlit:
       streamlit run app.py
5. Open the local URL shown in the terminal.

TEST CSV FORMAT
---------------
CSV must have 30 numeric feature columns matching the training dataset and one
diagnosis target column. diagnosis should be encoded as 0 (benign) or 1 (malignant).
The app drops optional id and Unnamed: 32 columns if present.
For the test dataset generated in the notebook, target column diagnosis is present.

GITHUB REPOSITORY
-----------------
Keep app.py, requirements.txt, README.txt, Data/, and model/ at the repository root.
Commit and push these files. Include the notebook/source code as requested by the
assignment. Check that the repository's model artifact is within GitHub's file size
limits; if it exceeds them, use Git LFS or an approved hosting approach.

STREAMLIT COMMUNITY CLOUD DEPLOYMENT
------------------------------------
1. Push the project to GitHub.
2. Open Streamlit Community Cloud and create a new app.
3. Choose the repository and branch.
4. Set app entry point to app.py at repository root.
5. Deploy. The Cloud environment installs packages from requirements.txt.

MODEL COMPATIBILITY / SECURITY
------------------------------
The combined model file was serialized with scikit-learn 1.9.0. The requirements
file pins scikit-learn 1.9.0 to match the training artifact. Serialized joblib
files can execute code during loading: only load trusted model files.

ASSIGNMENT README ITEMS TO FILL IN
----------------------------------
Problem statement:
[Explain the classification task.]

Dataset description:
[State source, number of instances/features and target classes.]

GitHub Repository Link:
[Paste the repository URL.]

Model metric comparison and observations:
[Use measured output from the notebook/app. Explain model performance.]

Overall winner:
[Select and justify based on the calculated metrics.]

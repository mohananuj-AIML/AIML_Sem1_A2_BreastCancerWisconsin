# Breast Cancer Classification - ML Assignment 2

## a. Problem statement
Create an application on streamlit for Machine learning assignment2. All the files should be uploaded to github, and streamlit application should directly connect to github to start the application. Application should compare five classification algorithms for predicting whether a breast tumor is benign or malignant using Kaggles Breast Cancer Wisconsin (Diagnostic) Data Set.

## b. Dataset description
Source: UCI Breast Cancer Wisconsin (Diagnostic) available on Kaggle. The dataset has 569 rows, 33 numerical features, and no missing values. The label is recoded to 0 = benign, 1 = malignant. Training dataset contains 455 data whereas testing dataset contains 114 data. 

## c. GitHub Repository Link
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download

### Model Evaluation Summary

| ML Model Name          | Accuracy |    AUC | Precision | Recall |     F1 |    MCC |
|:-----------------------|---------:|-------:|----------:|-------:|-------:|-------:|
| Logistic Regression    |   0.9825 |  0.9977 |     1.0000 | 0.9524 | 0.9756 | 0.9626 |
| Decision Tree          |   0.9561 |  0.9454 |     0.9744 | 0.9048 | 0.9383 | 0.9058 |
| KNN                    |   0.9386 |  0.9825 |     0.9730 | 0.8571 | 0.9114 | 0.8688 |
| Naive Bayes            |   0.9386 |  0.9934 |     1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine |   0.9737 |  0.9947 |     1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Optimized Hyperparameters & Confusion Matrices
* **Logistic Regression**
  * **Best Params:** `{'C': 0.1, 'max_iter': 100}`
  * **Confusion Matrix:** 
    ```text
    [[72  0]
     [ 2 40]]
    ```
* **Decision Tree**
  * **Best Params:** `{'criterion': 'entropy', 'max_depth': 10}`
  * **Confusion Matrix:** 
    ```text
    [[71  1]
     [ 4 38]]
    ```
* **KNN**
  * **Best Params:** `{'n_neighbors': 3, 'weights': 'uniform'}`
  * **Confusion Matrix:** 
    ```text
    [[71  1]
     [ 6 36]]
    ```
* **Naive Bayes**
  * **Best Params:** `{'var_smoothing': 1e-09}`
  * **Confusion Matrix:** 
    ```text
    [[72  0]
     [ 7 35]]
    ```
* **Support Vector Machine**
  * **Best Params:** `{'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}`
  * **Confusion Matrix:** 
    ```text
    [[72  0]
     [ 3 39]]
    ```

### Observations about model performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | **Logistic Regression** achieved the highest overall performance on the test set, with an outstanding accuracy of **98.25%** and a top F1-score of **97.56%**. It achieved a perfect Precision of **100%**, meaning it yielded zero false positives (no benign cases were incorrectly classified as malignant). It also showed the highest recall (**95.24%**), misclassifying only 2 malignant samples as benign. |
| **Decision Tree** | Optimized using the entropy criterion, the **Decision Tree** showed strong performance, achieving an accuracy of **95.61%** and an F1-score of **93.83%**. While highly efficient, it missed 4 actual malignant cases (Recall: 90.48%) and had 1 false positive instance. |
| **KNN** | **KNN** yielded an accuracy of **93.86%** and an F1-score of **91.14%** using a neighborhood size of 3. Feature scaling applied via the pipeline was crucial to its layout, though its recall (**85.71%**) trailed behind Logistic Regression and SVM. |
| **Naive Bayes** | **Gaussian Naive Bayes** produced an accuracy of **93.86%** and an F1-score of **90.91%**. While it yielded a perfect Precision score (**100%**), its lower recall score of **83.33%** (missing 7 malignant cases) makes it less ideal for this clinical context where minimizing false negatives is critical. |
| **Support Vector Machine** | The **SVM** model utilizing an RBF kernel proved highly reliable, achieving the second-best accuracy of **97.37%** and an F1-score of **96.30%**. Like Logistic Regression, it maintained a perfect Precision rate of **100%** and limited misclassified malignant cases down to just 3. |

### Overall winner for this dataset
After comparing all five optimized classification models, **Logistic Regression** emerged as the overall winner for this dataset. It achieved the highest **Accuracy (98.25%)**, highest **F1-Score (0.9756)**, highest **MCC (0.9626)**, and highest **Recall (95.24%)** while completely avoiding any false-positive classifications (100% Precision). 

**Overall Winner: Logistic Regression**

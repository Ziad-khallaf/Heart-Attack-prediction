# Heart-Attack-prediction Summary
Heart Attack Project with Recall 92.3 % using XGBClassifier and Accuracy 87 % using Random Forest Classifier, 
a project that uses historical data to feed a machine learning classification model in order to prdict the likelihood of heart attacks 


# Heart Attack Risk Prediction: A Machine Learning Approach

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Recall (XGBClassifier)](https://img.shields.io/badge/Recall-92.3%25-success?style=flat-square)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
[![Accuracy (RandomForestClassifier)](https://img.shields.io/badge/Accuracy-87%25-blueviolet?style=flat-square)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
[![Kaggle Dataset](https://img.shields.io/badge/Data-Kaggle-informational?style=flat-square&logo=kaggle)](https://www.kaggle.com/)

## Overview

This project focuses on predicting the likelihood of a heart attack using machine learning classification models trained on historical health data. The primary goal is to develop a system that can identify individuals at higher risk, enabling early intervention and potentially improving health outcomes.

The analysis pipeline culminates in a **Recall of 92.3%** achieved with the **XGBoost Classifier**, indicating a strong ability to identify positive cases (individuals who will experience a heart attack). Additionally, the **Random Forest Classifier** demonstrates a robust **Accuracy of 87%** for overall prediction performance.

## Data Source

The dataset utilized in this project is a single CSV file obtained from Kaggle.com. This file contains a collection of medical features and a target variable indicating whether a patient experienced a heart attack.

## Key Findings

* **Effective Prediction:** Machine learning models can effectively learn patterns from historical data to predict heart attack risk.
* **High Sensitivity:** The XGBoost Classifier's high recall (92.3%) is particularly important in a medical context, as it minimizes the chance of missing individuals who will have a heart attack (reducing false negatives).
* **Reliable Overall Performance:** The Random Forest Classifier's accuracy (87%) suggests a good balance between correctly identifying both positive and negative cases.
* **Potential for Early Intervention:** The developed model can serve as a valuable tool for healthcare professionals in assessing patient risk and implementing preventive measures.

## How to Use

1.  **Clone the Repository:**
2.  **Place the Data File:** Ensure that the heart attack data CSV file (replace `heart_attack_data.csv` with the actual name if different) is located in the `data/` directory of the cloned repository. If the `data/` directory doesn't exist, create it.

3.  **Install Dependencies:** Install the necessary Python libraries. It's recommended to use pip:
    ```bash
    pip install pandas scikit-learn xgboost matplotlib seaborn
    ```

4.  **Run the Notebook:** Execute the Jupyter Notebook (`filename.ipynb`) to reproduce the data analysis, model training, and evaluation steps.
    ```bash
    jupyter notebook filename.ipynb
    ```

## Project Structure

Heart-Attack-prediction/
├── data/
│   └── heart_attack_data.csv  # The heart attack dataset CSV file
├── notebooks/
│   └── filename.ipynb         # Jupyter Notebook containing the analysis
├── README.md

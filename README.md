# "AI-Powered Lung Cancer Diagnosis Through Oncogene Analysis"

## Overview
This project leverages artificial intelligence to enhance early detection and diagnosis of NSCLC by integrating clinical, genetic, and radiological data. It employs advanced machine learning algorithms—SVM, Random Forest, and XGBoost—to identify key oncogenic patterns. Core processes include data preprocessing and model evaluation using standard performance metrics. A user-friendly web interface is also developed to support real-time clinical decision-making.

## Method

****Dataset****
The NSCLC Radiogenomics dataset is available at: [https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/](https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/)

### Data Preprocessing
The data preprocessing phase involved cleaning, transforming, and preparing the dataset for machine learning. Missing values were handled using both simple imputation (e.g., mode, zero-fill) and advanced techniques like KNNImputer. Categorical variables were encoded into numerical values, and irrelevant or redundant columns were dropped. Continuous features were normalized using MinMaxScaler to ensure consistent scale across variables. Additional steps included extracting tumor locations, converting unknowns to NaNs, and shuffling the data to prevent bias. Finally, important features were selected using a Random Forest model to optimize prediction of recurrence and its location.
### Model Training
During the model training and evaluation phase, three algorithms were explored: SVM, XGBoost, and Random Forest.
SVM achieved promising accuracy (~80%) using clinically and genetically relevant features, but was excluded due to scalability limitations.
XGBoost showed strong performance and handled missing data effectively, yet its complexity and tuning requirements made it less ideal for deployment.
Random Forest was selected as the final model for its balance of accuracy, efficiency, and interpretability.
The data was preprocessed through feature selection, imputation, scaling, and stratified splitting (80/20).
Random Forest demonstrated high training and testing accuracy, confirming its robustness in predicting recurrence and recurrence locatio
## Results
Training Accuracy: The model achieved 95.83%, showing excellent performance in capturing patterns within the training data.
Testing Accuracy: The model achieved 86.05%, reflecting strong generalization to unseen data.

## Usage
### Dataset Preparation
Basic preprocessing steps were performed, including feature selection, handling missing values, target encoding, normalization, and train-test splitting.
A Random Forest classifier was trained on the processed data to predict recurrence and its location.
The model achieved strong performance and was saved to be integrated later into the API.
`NSCLC Project.ipynb`
### API
This API, built with Flask, predicts lung cancer recurrence and its potential location using clinical and genetic inputs. It integrates a trained Random Forest model (rf_model.pkl) along with a scaler (min_scaler.pkl) and feature encoders. User data is submitted via a web form, preprocessed (encoded, filled, scaled), and passed to the model. The API returns recurrence probability and, if applicable, the likelihood of recurrence location (local, regional, distant). Results are shown dynamically on a result page. It provides accurate, real-time predictions and is scalable for future enhancements.
`py.py`

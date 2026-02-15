Mobile Price Classification using Machine Learning
a. Problem Statement

The objective of this project is to classify mobile phones into different price categories based on their technical specifications. This is a multi-class classification problem where the target variable price_range consists of four classes representing increasing price levels.

Six machine learning classification models were implemented and compared using multiple evaluation metrics to analyze their performance.

b. Dataset Description

The dataset used in this project is the Mobile Price Classification Dataset obtained from Kaggle (datasets/iabhishekofficial/mobile-price-classification/train.csv).

Total Instances: 2000

Total Features: 20

Target Variable: price_range

Number of Classes: 4 (0 = Low, 1 = Medium, 2 = High, 3 = Very High)

Feature Type: All numeric

Missing Values: None

The features represent technical specifications of mobile devices such as:

Battery power

RAM

Internal memory

Camera resolution

Pixel resolution

Screen dimensions

Connectivity options (3G, 4G, WiFi, Bluetooth)

The dataset is balanced across all four price categories.

c. Models Used and Performance Comparison

The following six classification models were implemented:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble - Bagging)

XGBoost (Ensemble - Boosting)

All models were evaluated using:

Accuracy

AUC Score (One-vs-Rest for multi-class)

Precision (Weighted)

Recall (Weighted)

F1 Score (Weighted)

Matthews Correlation Coefficient (MCC)

📊 Performance Comparison Table
ML Model	Accuracy	AUC (OvR)	Precision	Recall	F1 Score	MCC
Logistic Regression	0.9650	0.9987	0.9650	0.9650	0.9650	0.9534
Decision Tree	0.8300	0.8867	0.8319	0.8300	0.8302	0.7738
KNN	0.5000	0.7698	0.5211	0.5000	0.5054	0.3350
Naive Bayes	0.8100	0.9506	0.8113	0.8100	0.8105	0.7468
Random Forest	0.8775	0.9795	0.8776	0.8775	0.8774	0.8368
XGBoost	0.9350	0.9945	0.9355	0.9350	0.9350	0.9135
d. Observations on Model Performance
Logistic Regression

Logistic Regression achieved the highest performance with an accuracy of 96.5% and the highest MCC score. This suggests that the dataset exhibits strong linear separability among price categories.

Decision Tree

The Decision Tree classifier captured nonlinear relationships but showed lower generalization performance compared to ensemble methods.

K-Nearest Neighbors (KNN)

KNN performed poorly with 50% accuracy. This may be attributed to the curse of dimensionality and limitations of distance-based learning in higher-dimensional feature spaces.

Naive Bayes

Naive Bayes achieved moderate performance. The independence assumption between features likely reduced its effectiveness for this dataset.

Random Forest

Random Forest improved performance over the single Decision Tree due to ensemble averaging, reducing overfitting and enhancing stability.

XGBoost

XGBoost achieved strong performance with high AUC and MCC values. However, it slightly underperformed compared to Logistic Regression, indicating that the dataset may be predominantly linearly structured.

Deployment

The trained models were deployed using Streamlit Community Cloud.
The application allows:

CSV dataset upload

Model selection

Display of evaluation metrics

Confusion matrix visualization
Dataset used - datasets/iabhishekofficial/mobile-price-classification/train.csv

📊 Final Performance Table (Formatted Cleanly)
Model				Accuracy	AUC (OvR)	Precision	Recall	F1 Score	MCC
Logistic Regression	0.9650		0.9987		0.9650		0.9650	0.9650		0.9534
Decision Tree		0.8300		0.8867		0.8319		0.8300	0.8302		0.7738
KNN					0.5000		0.7698		0.5211		0.5000	0.5054		0.3350
Naive Bayes			0.8100		0.9506		0.8113		0.8100	0.8105		0.7468
Random Forest		0.8775		0.9795		0.8776		0.8775	0.8774		0.8368
XGBoost				0.9350		0.9945		0.9355		0.9350	0.9350		0.9135

OBSERVATIONS - 

Logistic Regression

Logistic Regression achieved the highest performance among all models with an accuracy of 96.5% and an MCC score of 0.953. This indicates that the dataset is largely linearly separable. Features such as RAM, battery power, and pixel resolution strongly influence the price range in a near-linear manner, making logistic regression highly effective.

Decision Tree

The Decision Tree classifier achieved an accuracy of 83%. While it captures nonlinear relationships, it may suffer from overfitting and instability compared to ensemble methods. Its performance is significantly lower than Logistic Regression and XGBoost.

K-Nearest Neighbors (KNN)

KNN performed poorly with an accuracy of 50%. Despite feature scaling, distance-based methods struggle in higher-dimensional spaces due to the curse of dimensionality. This highlights the limitations of instance-based learning for this dataset.

Naive Bayes

Naive Bayes achieved 81% accuracy. Although computationally efficient, the assumption of feature independence does not fully hold in this dataset, reducing its performance compared to other models.

Random Forest

Random Forest improved performance over a single Decision Tree, achieving 87.75% accuracy. Ensemble averaging reduces overfitting and improves generalization, demonstrating the benefit of bagging techniques.

XGBoost

XGBoost achieved 93.5% accuracy with a very high AUC score of 0.994. As a boosting-based ensemble method, it effectively captures complex feature interactions. However, it slightly underperformed compared to Logistic Regression, indicating that the dataset may be more linearly structured than expected.
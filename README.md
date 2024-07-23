Predictive Modeling for URL Classification

Objective

The primary goal of this project is to evaluate and compare the performance of two machine learning algorithms—Logistic Regression and Decision Tree—in classifying URLs. The evaluation is performed using Weights & Biases (W&B) to track and compare the models' accuracy and performance metrics.

Project Overview

This project involves the following key components:

1. Data Preprocessing: 
   Cleaning and transforming the dataset to handle missing values, non-numeric values, and ensure the consistency of features.
   Encoding categorical variables and ensuring that the features used in training and testing phases match.

2. Model Training and Evaluation:
   Implementing and training Logistic Regression and Decision Tree classifiers.
   Using Weights & Biases to track experiments, visualize metrics, and compare model performance.

3. Model Comparison:
   Evaluating and comparing the accuracy, precision, recall, and other performance metrics of both classifiers.
   Analyzing the results to determine which model performs better for the given dataset.

Installation

To set up the project environment, ensure you have the following packages installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `graphviz`
- `dtreeviz`
- `wandb`

You can install these packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib graphviz dtreeviz wandb
```

Data

The dataset consists of URLs and various features related to their properties. The target variable is a binary classification indicating whether a URL is phishing or legitimate.

Steps

1. Data Loading:
   - Load the dataset and separate features and target variable.

2. Data Cleaning:
   - Handle missing values and non-numeric values.
   - Ensure consistency in feature names and types between training and test datasets.

3. Model Training:
   - Train Logistic Regression and Decision Tree models using the cleaned data.

4. Experiment Tracking:
   - Use Weights & Biases to track training runs and record performance metrics.

5. Model Evaluation:
   - Evaluate and compare the performance of both models using accuracy, precision, recall, and other relevant metrics.

6. Visualization:
   - Generate and visualize the decision tree to understand model decision boundaries.

Example Code

Here is an example of how to train and evaluate the models:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import wandb

Initialize W&B
wandb.init(project='url-classification')

Load and preprocess data
data = pd.read_csv('data.csv')
X = data.drop('Result', axis=1)
y = data['Result']

Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Train Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

Train Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_test)
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)

Log metrics to W&B
wandb.log({"Logistic Regression Accuracy": accuracy_log_reg})
wandb.log({"Decision Tree Accuracy": accuracy_decision_tree})

print(f"Logistic Regression Accuracy: {accuracy_log_reg}")
print(f"Decision Tree Accuracy: {accuracy_decision_tree}")
```

Results

- Logistic Regression Accuracy: [Value]
- Decision Tree Accuracy: (0.5)

The performance metrics indicate which model is more accurate for the given dataset.

Conclusion

Based on the experiments tracked via Weights & Biases, the project provides insights into the performance of Logistic Regression and Decision Tree classifiers. The analysis helps in determining the best model for URL classification tasks.




# Heart Disease Prediction using Machine Learning

## About the Project

This is a machine learning project for predicting heart disease based on a patient's medical information.

The project takes different health-related inputs such as age, sex, chest pain type, cholesterol, maximum heart rate, and other medical values. These values are processed and given to machine learning models to predict whether the patient has heart disease.

I have used different machine learning algorithms in this project and compared their performance before selecting a model for the final prediction.

---

## Dataset

The dataset used in this project is `heart.csv`.

It contains 13 input features and one target column.

The input columns are:

* age
* sex
* cp
* trestbps
* chol
* fbs
* restecg
* thalach
* exang
* oldpeak
* slope
* ca
* thal

The `target` column is used as the output of the model.

---

## Project Workflow

The basic workflow of this project is:

1. Load the dataset
2. Separate input and output columns
3. Split the data into training and testing data
4. Perform Yeo-Johnson transformation
5. Handle outliers
6. Perform feature selection
7. Balance the training data using SMOTE
8. Scale the data
9. Train different machine learning models
10. Check the performance of the models
11. Perform hyperparameter tuning
12. Save the trained model
13. Use the saved model for prediction

---

## Data Preprocessing

Before training the models, the data needs to be prepared properly.

### Yeo-Johnson Transformation

Yeo-Johnson transformation is used to transform the numerical features.

This helps to make the data distribution more suitable for the machine learning models.

The important thing here is that the transformation parameters learned from the training data should also be used when processing new data.

### Outlier Treatment

I have used the IQR method to identify and handle outliers.

The IQR is calculated using:

```text
IQR = Q3 - Q1
```

The upper and lower limits are calculated as:

```text
Upper Limit = Q3 + 1.5 × IQR

Lower Limit = Q1 - 1.5 × IQR
```

Values outside these limits are trimmed to the respective boundary.

---

## Feature Selection

Feature selection is used to remove features that are not useful for the model.

The project uses the following methods:

### Constant Features

Features with zero variance are removed using `VarianceThreshold`.

### Quasi-Constant Features

Features with very low variance are also removed.

### Hypothesis Testing

Pearson correlation and p-values are used to check the relationship between the features and the target.

Features with a p-value greater than `0.05` are removed in the current implementation.

---

## Handling Imbalanced Data

SMOTE is used to balance the training data.

SMOTE creates synthetic samples for the minority class instead of simply duplicating existing rows.

This is done only on the training data.

```text
Training Data
      ↓
    SMOTE
      ↓
Balanced Training Data
```

---

## Feature Scaling

After feature selection and SMOTE, the data is scaled using `StandardScaler`.

Standardization uses the formula:

```text
z = (x - mean) / standard deviation
```

The scaler is fitted using the training data and then used to transform the test data.

The scaler is saved so that it can also be used when making predictions on new data.

---

## Machine Learning Models

I have used several classification algorithms in this project:

* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes
* Logistic Regression
* Decision Tree
* Random Forest
* AdaBoost
* Gradient Boosting
* XGBoost

Each model is trained using the processed training data.

---

## Model Evaluation

The models are evaluated using different metrics.

### Accuracy

Accuracy tells us how many predictions were correct out of the total predictions.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Confusion Matrix

The confusion matrix shows:

* True Positive
* True Negative
* False Positive
* False Negative

### Classification Report

The classification report gives:

* Precision
* Recall
* F1-score
* Support

### ROC Curve

ROC curves are also generated to compare the classification performance of the models.

For the ROC curve, the model probability is used instead of directly using the predicted class.

For example:

```python
model.predict_proba(X_test)[:, 1]
```

---

## Hyperparameter Tuning

After testing the models, Gaussian Naive Bayes is further tuned using `GridSearchCV`.

The parameter being tuned is:

```text
var_smoothing
```

Different values are tested and the best model found by `GridSearchCV` is used.

---

## Saving the Model

The trained model is saved using Pickle.

```text
trained_model.pkl
```

The scaler is also saved:

```text
scaled_values.pkl
```

These files can be loaded later without training the model again.

---

## Prediction

For prediction, the user provides the original patient information.

The input contains values such as:

```text
age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
```

The new data should go through the same preprocessing steps that were used during training.

```text
New Patient Data
       ↓
Preprocessing
       ↓
Feature Selection
       ↓
Scaling
       ↓
Trained Model
       ↓
Prediction
```

The important point is that the user should enter the original patient values. The values should not already be Yeo-Johnson transformed.

---

## Technologies Used

### Programming Language

Python

### Libraries

* NumPy
* Pandas
* SciPy
* Scikit-learn
* Imbalanced-learn
* XGBoost
* Matplotlib

Pickle is also used for saving the trained model and scaler.

---

## Project Files

```text
heart.csv
main.py
var_and_out.py
feature_selection.py
selecting_best_model.py
hyperparameter_tuning.py
log_code.py
```

The project also generates:

```text
trained_model.pkl
scaled_values.pkl
main.log
feature_selection.log
```

---

## How to Run the Project

First install the required libraries:

```bash
pip install numpy pandas scipy scikit-learn imbalanced-learn xgboost matplotlib
```

Then run:

```bash
python main.py
```

The program will load the dataset, preprocess the data, train the models, evaluate them, perform hyperparameter tuning, and save the required model files.

---

## Important Note

The preprocessing used during training and prediction should be the same.

For example, if Yeo-Johnson transformation is applied during training, the same transformation parameters should be used when processing new patient data.

The same applies to:

* Feature selection
* Outlier treatment
* Scaling

This is important because the model expects the prediction data in the same format as the data it was trained on.

---

## Conclusion

This project helped me understand the complete machine learning workflow, starting from data preprocessing and feature selection to model training, evaluation, hyperparameter tuning, and prediction.

It also demonstrates how a trained machine learning model and scaler can be saved and reused for making predictions on new data.

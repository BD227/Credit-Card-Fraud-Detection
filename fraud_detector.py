import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset from the csv file using pandas
# best way is to mount the drive on colab and 
# copy the path for the csv file
data = pd.read_csv("creditcard.csv")

# Determine number of fraud cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

# dividing the X and the Y from the dataset
X = data.drop(['Class'], axis = 1)
Y = data["Class"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    "Random Forest (Unscaled)": RandomForestClassifier(class_weight='balanced'),
    "Logistic Regression (Scaled)": LogisticRegression(class_weight='balanced'),
    "Gradient Boosting (Unscaled)": GradientBoostingClassifier()
}

# Define K-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X_scaled if 'Scaled' in name else X, Y, cv=kfold, scoring='f1')
    print(f"{name}: Average F1 Score = {scores.mean():.4f}, Std Dev = {scores.std():.4f}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset from the csv file using pandas
data = pd.read_csv("creditcard.csv")

# Determine number of fraud cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

# Dividing the X and Y from the dataset
X = data.drop(['Class'], axis = 1)
Y = data["Class"]

# Define models to be compared
models = {
    "Random Forest (Unscaled)": RandomForestClassifier(class_weight='balanced'),
    "Logistic Regression (Scaled)": LogisticRegression(class_weight='balanced'),
    "Gradient Boosting (Unscaled)": GradientBoostingClassifier()
}

# Define K-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store F1 scores in a list for boxplot
scores_data = []

# Run cross-validation for each model and store the F1 scores
for name, model in models.items():
    scores = cross_val_score(model, X, Y, cv=kfold, scoring='f1')
    for score in scores:
        scores_data.append({'Model': name, 'F1 Score': score})

# Convert scores to DataFrame for easier plotting
scores_df = pd.DataFrame(scores_data)

# Plot the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Model', y='F1 Score', data=scores_df, palette="Set2")
plt.title('Model Performance Across Cross-Validation Folds (F1 Score)')
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

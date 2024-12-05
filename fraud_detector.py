import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("creditcard.csv")

# Determine number of fraud cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))
print('Fraud Percentage: {:.2f}%'.format(outlierFraction * 100))

# Separate features and labels
X = data.drop(columns=["Class"])
Y = data["Class"]

# Scale 'Time' and 'Amount' only, as other columns are already scaled
X["Time"] = StandardScaler().fit_transform(X["Time"].values.reshape(-1, 1))
X["Amount"] = StandardScaler().fit_transform(X["Amount"].values.reshape(-1, 1))

# Define models
models = {
    "Logistic Regression": LogisticRegression(class_weight=None, max_iter=1000, solver="liblinear"),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# Define cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define sampling techniques
sampling_techniques = {
    "No Sampling": (X, Y),
    "Undersampling": RandomUnderSampler(random_state=42).fit_resample(X, Y),
}

# Collect results for plotting
results = []

# Train and compare models
for sampling_name, (X_sampled, Y_sampled) in sampling_techniques.items():
    print(f"\n=== Sampling Technique: {sampling_name} ===")
    print(f"Class distribution: {Counter(Y_sampled)}")

    for name, model in models.items():
        print(f"Training {name} with {sampling_name}...")

        # Fit the model
        model.fit(X_sampled, Y_sampled)

        # Perform cross-validation
        scores = cross_val_score(model, X_sampled, Y_sampled, cv=kfold, scoring="f1")

        # Store results
        results.append({
            "Model": name,
            "Sampling": sampling_name,
            "Average F1 Score": scores.mean(),
            "F1 Std Dev": scores.std(),
        })

        print(f"{name} ({sampling_name}): Average F1 Score = {scores.mean():.4f}, Std Dev = {scores.std():.4f}")

# Convert results to a DataFrame for plotting
results_df = pd.DataFrame(results)

# Plotting results
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="Model", y="Average F1 Score", hue="Sampling")
plt.title("Model Comparison Across Sampling Techniques")
plt.ylabel("Average F1 Score")
plt.xlabel("Model")
plt.legend(title="Sampling Technique")
plt.tight_layout()
plt.show()

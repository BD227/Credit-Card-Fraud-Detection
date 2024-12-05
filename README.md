# Credit-Card-Fraud-Detection
Credit card fraud detection using Logistic Regression, Random Forest, and Gradient Boosting. This project compares each technique and demonstrates the importance of properly handling imbalanced data sets.

## Results
=== Sampling Technique: No Sampling ===
Class distribution: Counter({0: 284315, 1: 492})
Training Logistic Regression with No Sampling...
Logistic Regression (No Sampling): Average F1 Score = 0.7290, Std Dev = 0.0324
Training Random Forest with No Sampling...
Random Forest (No Sampling): Average F1 Score = 0.8510, Std Dev = 0.0208
Training Gradient Boosting with No Sampling...
Gradient Boosting (No Sampling): Average F1 Score = 0.6615, Std Dev = 0.1608

=== Sampling Technique: Undersampling ===
Class distribution: Counter({0: 492, 1: 492})
Training Logistic Regression with Undersampling...
Logistic Regression (Undersampling): Average F1 Score = 0.9397, Std Dev = 0.0182
Training Random Forest with Undersampling...
Random Forest (Undersampling): Average F1 Score = 0.9337, Std Dev = 0.0172
Training Gradient Boosting with Undersampling...
Gradient Boosting (Undersampling): Average F1 Score = 0.9404, Std Dev = 0.0169

## Details
To begin, we apply a standard scalar to time and amount to match the rest of the data.

X["Time"] = StandardScaler().fit_transform(X["Time"].values.reshape(-1, 1))
X["Amount"] = StandardScaler().fit_transform(X["Amount"].values.reshape(-1, 1))

If we look at the number of fraudulant cases we see these only make up a small portion of the entire dataset. There are only 492 frauds out of 284,807 transactions (0.17%). The data is highly imbalanced. To compensate we need to balance the data using sampling. In this case we will use undersampling.

RandomUnderSampler(random_state=42).fit_resample(X, Y)

This reduces the size of the majority class, in this case valid transactions, to match the size of the minority, the fraudulant cases. This is especially important since we care more about identifying the fraudulant cases.

## Resources

The data set contains transactions made by credit cards in September 2023 by European cardholders. The data set is available on Kaggle here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

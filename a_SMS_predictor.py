# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Load Dataset and since old data we use latin-1

df = pd.read_csv("spam.csv", encoding="latin-1")

# Display Dataset

print(df.head())

# Dataset Information

print(df.info())

# Check Missing Values

print(df.isnull().sum())

# Remove Unnecessary Columns (the ,,, in dataset)

df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)

# Rename Columns

df.rename(columns={
    "v1": "label",
    "v2": "message"
}, inplace=True)

# Remove Duplicate Rows

df.drop_duplicates(inplace=True)

# Encode Target Variable

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Split Features and Target

X = df["message"]
y = df["label"]

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert Text into TF-IDF Features giving imp rating to all phrases

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Naive Bayes Model

model = MultinomialNB()

model.fit(X_train, y_train)

# Predictions

y_pred = model.predict(X_test)

# Prediction Probabilities

y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation Metrics

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
print("Recall :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC Score :", roc_auc_score(y_test, y_prob))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# ROC Curve

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("ROC Curve")
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# Class Distribution

sns.countplot(data=df, x="label")

plt.xticks([0, 1], ["Ham", "Spam"])
plt.title("Class Distribution")
plt.xlabel("Message Type")
plt.ylabel("Count")

plt.savefig("class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Predict Custom SMS

message = input("Enter an SMS: ")

message_vector = vectorizer.transform([message])

prediction = model.predict(message_vector)

probability = model.predict_proba(message_vector)

if prediction[0] == 1:
    print("\nPrediction : Spam")
else:
    print("\nPrediction : Ham")

print(f"Spam Probability : {probability[0][1]:.4f}")
print(f"Ham Probability  : {probability[0][0]:.4f}")


# Accuracy : 0.9506769825918762
# Precision : 1.0
# Recall : 0.6106870229007634
# F1 Score : 0.7582938388625592
# ROC-AUC Score : 0.9869392102660344          For references
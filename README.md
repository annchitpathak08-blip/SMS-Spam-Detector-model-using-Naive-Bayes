# SMS Spam Detection using Naive Bayes

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using **TF-IDF Vectorization** and the **Multinomial Naive Bayes** algorithm.

This project demonstrates a complete Natural Language Processing (NLP) workflow, including data preprocessing, feature extraction, model training, evaluation, visualization, and real-time message prediction. It is designed to help beginners understand how classical machine learning techniques can be applied to text classification problems.

---

# Project Overview

Spam detection is one of the most common applications of Natural Language Processing. The objective of this project is to build a classifier capable of identifying whether an incoming SMS message is legitimate or spam.

The model learns word patterns from thousands of labeled SMS messages and predicts the category of unseen messages using probability-based classification.

---

# Features

* Complete end-to-end NLP pipeline
* Data cleaning and preprocessing
* Duplicate removal
* Label encoding
* TF-IDF feature extraction
* Multinomial Naive Bayes classifier
* Performance evaluation using multiple metrics
* Confusion Matrix visualization
* ROC Curve visualization
* Class Distribution visualization
* Interactive prediction for custom SMS messages

---

# Dataset

The project uses the **SMS Spam Collection Dataset**, containing more than **5,500** labeled SMS messages.

Classes:

* Ham (Legitimate Messages)
* Spam (Unwanted or Promotional Messages)

---

# Project Structure

```text
```text
SMS-Spam-Classifier/
│
├── images/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── spam.csv
├── sms_spam_classifier.py
├── requirements.txt
└── README.md
```

```

---

# Machine Learning Workflow

```text
Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
Label Encoding
        │
        ▼
Train-Test Split
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Multinomial Naive Bayes
        │
        ▼
Prediction
        │
        ▼
Model Evaluation
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

# Concepts Covered

This project demonstrates the following machine learning and NLP concepts:

* Data Cleaning
* Missing Value Analysis
* Duplicate Removal
* Label Encoding
* Train-Test Split
* TF-IDF Vectorization
* Text Feature Engineering
* Multinomial Naive Bayes
* Model Evaluation
* Probability-Based Classification

---

# Model Evaluation

The model is evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Classification Report
* Confusion Matrix

Using multiple evaluation metrics provides a better understanding of model performance, particularly because the SMS dataset is imbalanced.

---

# Visualizations

## Class Distribution

Shows the distribution of Ham and Spam messages within the dataset.

![Class Distribution](class_distribution.png)

---

## Confusion Matrix

Displays the number of correct and incorrect predictions made by the classifier.

![Confusion Matrix](confusion_matrix.png)

---

## ROC Curve

Illustrates the model's ability to distinguish between Spam and Ham across different classification thresholds.

![ROC Curve](roc_curve.png)

---

# Custom SMS Prediction

After training, the project allows users to enter their own SMS message directly from the terminal.

Example:

```text
Enter an SMS:
Congratulations! You have won a FREE vacation. Click here to claim your prize.

Prediction : Spam

Spam Probability : 0.9984
Ham Probability  : 0.0016
```

---

---

# Results

The trained model successfully classifies SMS messages as Spam or Ham using probabilistic classification.

The project also demonstrates how TF-IDF converts text into numerical features before passing them to the Naive Bayes classifier. Performance is evaluated using multiple classification metrics and visualizations to provide a complete understanding of the model's behavior.

---

# Future Improvements

Possible enhancements include:

* Compare TF-IDF with CountVectorizer
* Hyperparameter tuning for Naive Bayes
* Character-level n-grams for improved typo handling
* Stemming and Lemmatization
* Streamlit or Flask web application
* Comparison with Logistic Regression, Support Vector Machine, and Random Forest
* Cross-validation for more robust evaluation

---

# Requirements

All required Python libraries are listed in the **requirements.txt** file.

Install them using:

```bash
pip install -r requirements.txt
```

---

# Acknowledgements

This project was built as part of my machine learning learning journey to strengthen my understanding of Natural Language Processing, text preprocessing, feature engineering, and probabilistic classification using Scikit-learn.

---

# Contributing

Contributions, suggestions, and improvements are always welcome. Feel free to fork the repository, open an issue, or submit a pull request.

---

# License

This project is intended for educational and learning purposes.

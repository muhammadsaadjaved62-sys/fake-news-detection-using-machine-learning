# Project Report: Fake News Detection Using Machine Learning

## 1. Introduction
Fake news can spread quickly through online platforms. This project demonstrates a basic machine learning approach for classifying news text into two categories: Fake and Real.

## 2. Objective
To build a simple text-classification system using Python and machine learning, and provide a user interface where a user can enter news text and view a prediction.

## 3. Tools and Technologies
- Python for programming
- pandas for reading and preparing CSV data
- scikit-learn for machine learning
- TF-IDF for text feature extraction
- Logistic Regression for classification
- Streamlit for the user interface

## 4. Methodology
1. Load fake and real news examples from separate CSV files.
2. Assign labels: 0 for Fake and 1 for Real.
3. Clean the text by converting it to lowercase and removing URLs and non-letter characters.
4. Split the data into training and testing sets.
5. Apply TF-IDF vectorization.
6. Train a Logistic Regression classifier.
7. Evaluate the model on the held-out test set.
8. Save the model and vectorizer for use in the Streamlit application.

## 5. Evaluation
Run `python train_model.py` to generate the actual accuracy, classification report, and confusion matrix. Insert the measured results here after running the project. Do not claim an accuracy value before testing.

## 6. Application
The Streamlit application accepts a headline or article, applies the same text cleaning and TF-IDF transformation, and displays the predicted class and model scores.

## 7. Limitations
The model learns statistical patterns from its dataset. It does not search the internet or verify facts. Results can be wrong when the input differs from the training examples or contains satire, opinion, or unfamiliar topics.

## 8. Conclusion
This project provides practical experience with text preprocessing, feature extraction, supervised learning, model evaluation, and a simple machine learning interface.

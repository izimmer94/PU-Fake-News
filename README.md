# Positive-Unlabeled (PU) Fake News Problem
Authors: Atin Kolli, Ethan Wong, Ian Zimmermann

---
This project aims to analyze news articles to predict whether they are fake or real. However, instead of using the traditional approach with fully labeled datasets, positive samples will be treated as unlabeled creating a positive-unlabeled (PU) learning scenario. This method adds an extra layer of difficulty in the classification process that we hope to tackle in this project.

## PU Learning
PU learning is a machine learning problem where true positives are known, but true negatives are unknown. This was achieved in the original dataset by 'blurring' the labels; half of the original positive labels were transformed to negative labels. This results in the ML models being trained on true positives and unknown labels. 

## Ensemble Learning
### Preprocessing
The textual data of article title and bodies were transformed converted into quantitative features through two main approaches. Vader sentiment for tone of the text and TF-IDF with dimensionality reduction which is an importance weighted vector similar to bag of words.

### Creating model
Three main base classifiers were trained logistic regression, SVM, naive bayes. Logistic regression and SVM had modified cutoffs of 0.8. Final predictions were based on a simple majority vote of the three base classifiers. 

## Results
The ensemble model performed better compared to the individual accuracies of each of the base classifiers. Using the original true labels of the dataset, the ensemble model achieved an accuracy of 84%. 
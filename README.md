# Facial Expression Recognition using KNN and LBP

This repository contains a facial expression recognition system that uses **K-Nearest Neighbors (KNN)** as the classification model and **Local Binary Patterns (LBP)** as the feature extraction technique. The goal is to classify facial expressions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Overview

The code loads grayscale images of faces from the `train` and `test` folders, applies LBP to extract texture features, and then uses KNN to classify the expressions. A grid search over several parameters for both LBP and KNN is performed to find the optimal settings, aiming to improve the model’s accuracy.

## Files

- `main.py`: The main script that loads the data, performs feature extraction using LBP, trains the KNN classifier, and evaluates the model.
- `knn_model_optimized_final.pkl`: The saved model file containing the trained KNN classifier with the best-found parameters.
- `classification_report_optimized_final.txt`: A text file containing the accuracy and detailed classification report of the final model.
- `README.md`: This file, explaining the structure of the repository and the functionality of each component.

## How It Works

1. **Feature Extraction**: Each image is processed using LBP, a texture descriptor that captures local patterns in the image. The `radius` and `n_points` parameters of LBP are optimized to capture the most relevant details.

2. **Classification with KNN**: The K-Nearest Neighbors algorithm is used to classify the facial expressions. A grid search is performed over multiple values for `n_neighbors` and different distance metrics (Euclidean, Manhattan, and Chebyshev) to identify the best configuration.

3. **Evaluation**: The model is evaluated on the test dataset, and metrics such as accuracy, precision, recall, and F1-score are calculated and saved in a report file.

## Results

The grid search identifies the optimal parameters for both LBP and KNN, which are then used to train the final model. The results, including accuracy and the classification report, are saved for analysis.

## Usage

1. Place your training and testing data in the `archive/train` and `archive/test` folders, respectively.
2. Run `main.py` to perform feature extraction, train the model, and evaluate it.
3. The final trained model will be saved as `knn_model_optimized_final.pkl`, and the evaluation results will be saved in `classification_report_optimized_final.txt`.

## Requirements

- Python 3.6 or later
- OpenCV
- scikit-image
- scikit-learn
- tqdm
- joblib


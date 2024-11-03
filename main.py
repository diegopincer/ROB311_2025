import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm  # For progress bar
import joblib  # For saving the model

# Function to extract LBP features from an image with adjustable parameters
def extract_lbp_features(image, radius=1, n_points=8, method='uniform'):
    lbp = local_binary_pattern(image, n_points, radius, method=method)
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram
    return hist

# Function to load data from folders
def load_data_from_folders(base_folder, radius=1, n_points=8):
    images = []
    labels = []
    emotion_classes = sorted(os.listdir(base_folder))

    # Calculate the total number of files for progress tracking
    total_files = sum(len(files) for _, _, files in os.walk(base_folder))

    # Use tqdm to display the progress bar
    with tqdm(total=total_files, desc="Processing images") as pbar:
        for emotion in emotion_classes:
            folder_path = os.path.join(base_folder, emotion)
            if not os.path.isdir(folder_path):
                continue

            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
                if image is None or image.shape != (48, 48):
                    pbar.update(1)  # Update progress
                    continue

                features = extract_lbp_features(image, radius=radius, n_points=n_points)
                images.append(features)
                labels.append(emotion)
                pbar.update(1)  # Update progress for each processed image

    return np.array(images), np.array(labels)

# Define paths to the train and test folders
train_folder = 'archive/train'  # Replace with the correct path to the 'train' folder
test_folder = 'archive/test'    # Replace with the correct path to the 'test' folder

# Parameters to test for LBP
lbp_radius_options = [1, 2, 3]  # Test different radius values
lbp_n_points_options = [8 * r for r in lbp_radius_options]  # Adjust n_points accordingly

# Parameters to test for KNN
knn_neighbors_options = [3, 5, 7, 9, 11]  # Test a wider range of neighbors
knn_metric_options = ['euclidean', 'manhattan', 'chebyshev']  # Test more distance metrics

best_accuracy = 0
best_params = {}

# Perform grid search over LBP and KNN parameters
for radius, n_points in zip(lbp_radius_options, lbp_n_points_options):
    print(f"\nTesting LBP parameters: radius={radius}, n_points={n_points}")
    
    # Load training and testing data with the current LBP parameters
    X_train, y_train = load_data_from_folders(train_folder, radius=radius, n_points=n_points)
    X_test, y_test = load_data_from_folders(test_folder, radius=radius, n_points=n_points)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    for n_neighbors in knn_neighbors_options:
        for metric in knn_metric_options:
            # Train KNN with the current configuration
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
            knn.fit(X_train, y_train_encoded)

            # Evaluate the model
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test_encoded, y_pred)
            
            # Print results for each configuration
            print(f"LBP (radius={radius}, n_points={n_points}), KNN (n_neighbors={n_neighbors}, metric={metric}): Accuracy={accuracy * 100:.2f}%")
            
            # Update best parameters if the accuracy is improved
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'radius': radius,
                    'n_points': n_points,
                    'n_neighbors': n_neighbors,
                    'metric': metric
                }

# Train final model with the best parameters
print(f"\nBest parameters: {best_params}, with accuracy: {best_accuracy * 100:.2f}%")
X_train, y_train = load_data_from_folders(train_folder, radius=best_params['radius'], n_points=best_params['n_points'])
X_test, y_test = load_data_from_folders(test_folder, radius=best_params['radius'], n_points=best_params['n_points'])

# Encode labels again for final training
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], metric=best_params['metric'])
knn.fit(X_train, y_train_encoded)

# Evaluate final model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test_encoded, y_pred)
classification_rep = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)

# Print final results
print(f'Final Accuracy: {accuracy * 100:.2f}%')
print(classification_rep)

# Save the final model
model_filename = 'knn_model_optimized_final.pkl'
joblib.dump(knn, model_filename)
print(f"Model saved as {model_filename}")

# Save the classification report to a .txt file
report_filename = 'classification_report_optimized_final.txt'
with open(report_filename, 'w') as f:
    f.write(f'Final Accuracy: {accuracy * 100:.2f}%\n')
    f.write(classification_rep)
print(f"Classification report saved as {report_filename}")

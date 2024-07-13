import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA
from mne.decoding import CSP
from sklearn import svm, model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import os
import mne

# Define a function to read and preprocess EEG data from a GDF file
def read_data(path):
    # Read raw EEG data from GDF file, preload data and specify EOG channels
    raw = mne.io.read_raw_gdf(path, preload=True, eog=['EOG-left', 'EOG-central', 'EOG-right'])
    # Drop EOG channels from the data
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    # Set EEG reference
    raw.set_eeg_reference()
    # Extract events from annotations in the data
    events = mne.events_from_annotations(raw)
    # Create epochs based on events and specify event IDs of interest
    epochs = mne.Epochs(raw, events[0], event_id=[7, 8, 9, 10], on_missing='warn')
    # Extract labels (event IDs) from epochs
    labels = epochs.events[:, -1]
    # Extract features (EEG data) from epochs
    features = epochs.get_data()
    return features, labels

# File paths for each patient's data
file_paths = [
    f"D:\\hci V3\\data\\A01T.gdf",
    f"D:\\hci V3\\data\\A02T.gdf",
    f"D:\\hci V3\\data\\A03T.gdf",
    f"D:\\hci V3\\data\\A04T.gdf",
    f"D:\\hci V3\\data\\A05T.gdf",
    f"D:\\hci V3\\data\\A06T.gdf",
    f"D:\\hci V3\\data\\A07T.gdf",
    f"D:\\hci V3\\data\\A08T.gdf",
    f"D:\\hci V3\\data\\A09T.gdf"
]

# Initialize lists to store features and labels for all patients
features_list = []
labels_list = []

# Process data for each patient and append to the lists
for path in file_paths:
    features, labels = read_data(path)
    features_list.append(features)
    labels_list.append(labels)

# Combine data from all patients into single arrays
all_features = np.concatenate(features_list, axis=0)
all_labels = np.concatenate(labels_list, axis=0)

# Apply Independent Component Analysis (ICA) for artifact removal and feature extraction
ica = FastICA(n_components=22, tol=0.01, max_iter=5000, whiten='arbitrary-variance')
# Transform the data using ICA
ica_features = ica.fit_transform(all_features.reshape(all_features.shape[0], -1)).reshape(all_features.shape[0], 22, -1)

# Print the shape of the transformed features
print("Shape of ica_features:", ica_features.shape)

# Transpose data to match MNE's expected shape (samples, channels, time)
mne_signals = np.transpose(ica_features, (0, 2, 1))
# Normalize the data
mne_signals = (mne_signals - np.mean(mne_signals, axis=2, keepdims=True)) / np.std(mne_signals, axis=2, keepdims=True)
# Apply Common Spatial Pattern (CSP) for feature extraction
spatial_filter = CSP(n_components=12).fit(mne_signals, all_labels)
# Transform the data using CSP
mne_signals_reduced = spatial_filter.transform(mne_signals)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mne_signals_reduced, all_labels, train_size=0.7, test_size=0.3, random_state=90)

# Train a Support Vector Machine (SVM) with RBF kernel
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=75).fit(X_train, y_train)
# Save the trained SVM model
joblib.dump(rbf, 'svm_model.pkl', compress=9)

# Predict on the test set and calculate accuracy
poly_pred = rbf.predict(X_test)
poly_accuracy = accuracy_score(y_test, poly_pred)
print('Accuracy (RBF Kernel): ', "%.2f" % (poly_accuracy * 100))

# Display the confusion matrix
cm = confusion_matrix(y_test, poly_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

# Train a Random Forest Classifier
classifier = RandomForestClassifier(max_depth=26, random_state=25)
classifier.fit(X_train, y_train)
# Save the trained Random Forest model
joblib.dump(classifier, 'rf_model.pkl', compress=9)

# Predict on the test set and calculate accuracy
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy (Random Forest):", acc * 100, "%")

# Train a K-Nearest Neighbors (KNN) classifier
knn = KNeighborsClassifier(n_neighbors=80)
knn.fit(X_train, y_train)
# Save the trained KNN model
joblib.dump(knn, 'knn_model.pkl', compress=9)

# Predict on the test set and calculate accuracy
pred_knn = knn.predict(X_test)
acc = accuracy_score(y_test, pred_knn)
print("Accuracy (KNN):", acc * 100, "%")

# Display the confusion matrix
cm = confusion_matrix(y_test, pred_knn)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

# Define parameter distribution for Randomized Search CV
param_dist = {'n_estimators': randint(50, 1000), 'max_depth': randint(1, 20)}
# Initialize Random Forest Classifier
rf = RandomForestClassifier()
# Perform Randomized Search CV
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=75, cv=5)
rand_search.fit(X_train, y_train)

# Get the best model from Randomized Search CV
best_rf = rand_search.best_estimator_
# Save the best Random Forest model
joblib.dump(best_rf, 'best_rf_model.pkl', compress=9)

# Print the best hyperparameters
print('Best hyperparameters:', rand_search.best_params_)

# Predict on the test set using the best model and calculate accuracy
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (Best Random Forest):", accuracy)

# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

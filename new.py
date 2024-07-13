import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from mne.decoding import CSP
import tkinter as tk

# Function to load EEG data from CSV files
def load_eeg_data(csv_paths):
    data = []
    labels = []
    for path in csv_paths:
        df = pd.read_csv(path)
        # Extract EEG signal columns
        eeg_signals = df.filter(regex='EEG-').values
        data.append(eeg_signals)
        # Extract labels
        labels.extend(df['label'].values)
    return np.concatenate(data), np.array(labels)

# Define CSV file paths
csv_paths = ['E:/hci project/archive/BCICIV_2a_all_patients.csv']

# Load data
eeg_data, labels = load_eeg_data(csv_paths)

# Check if we have enough samples
if eeg_data.shape[0] < 2:
    raise ValueError("Not enough samples in the dataset. Ensure the dataset contains multiple samples.")

# Normalize data
eeg_data = (eeg_data - np.mean(eeg_data, axis=0)) / np.std(eeg_data, axis=0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(eeg_data, labels, test_size=0.2, random_state=42)

# Feature extraction using CSP
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
X_train_csp = csp.fit_transform(X_train, y_train)
X_test_csp = csp.transform(X_test)

# Train and evaluate classifier
clf = LinearDiscriminantAnalysis()
param_grid = {'solver': ['svd', 'lsqr', 'eigen'], 'shrinkage': [None, 'auto', 0.1, 0.5, 1]}
grid = GridSearchCV(clf, param_grid, cv=5)
grid.fit(X_train_csp, y_train)
best_clf = grid.best_estimator_
y_pred = best_clf.predict(X_test_csp)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# GUI Application
class BCIApp:
    def __init__(self, root, classifier, csp):
        self.root = root
        self.classifier = classifier
        self.csp = csp
        self.create_widgets()

    def create_widgets(self):
        self.root.title("Motor Imagery BCI Interface")
        self.root.geometry("300x200")
        activities = ['Sleep', 'Eat', 'Drink Water', 'Go to Bathroom']
        self.buttons = []
        for activity in activities:
            button = tk.Button(self.root, text=activity, command=lambda c=activity: self.on_choice(c))
            button.pack(pady=10)
            self.buttons.append(button)

    def on_choice(self, activity):
        print(f'Selected activity: {activity}')

root = tk.Tk()
app = BCIApp(root, best_clf, csp)
root.mainloop()

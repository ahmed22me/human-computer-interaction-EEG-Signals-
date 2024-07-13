import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load CSV data into DataFrame
df = pd.read_csv('E:/hci project/archive/BCICIV_2a_all_patients.csv')

# Extract relevant EEG channel names and label
eeg_channels = ['EEG-Fz', 'EEG-C3', 'EEG-Cz', 'EEG-C4', 'EEG-Pz']
X = df[eeg_channels].values  # EEG channel values
y = df['label'].values  # Motor imagery labels

# Bandpass filter settings
def bandpass_filter(data, lowcut, highcut, fs):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Apply bandpass filtering to EEG data
fs = 250  # Sampling rate (assuming 250 Hz)
lowcut = 8
highcut = 30
X_filtered = bandpass_filter(X.T, lowcut, highcut, fs).T

# Define epoch duration and sampling rate
epoch_duration = 4  # Epoch duration in seconds
epoch_samples = epoch_duration * fs  # Number of samples per epoch

# Initialize lists for epochs and labels
epochs = []
labels = []

# Iterate over DataFrame to extract epochs
for idx, row in df.iterrows():
    start_sample = int(row['time'] * fs)
    epoch_data = X_filtered[start_sample:start_sample + epoch_samples, :]
    
    # Ensure all epoch_data segments have consistent length (epoch_samples)
    if epoch_data.shape[0] == epoch_samples:
        epochs.append(epoch_data)
        labels.append(row['label'])

# Convert lists to numpy arrays
epochs = np.array(epochs)
labels = np.array(labels)

# Reduce number of epochs to avoid memory issues (example: use first 1000 epochs)
epochs = epochs[:1000]
labels = labels[:1000]

# Initialize CSP transformer with reduced components
n_components = 4  # Number of CSP components
csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)

# Apply CSP on epochs and labels
X_csp = csp.fit_transform(epochs, labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_csp, labels, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm = SVC(kernel='linear', C=1)

# Train SVM classifier
svm.fit(X_train, y_train)

# Predict on test set
y_pred = svm.predict(X_test)

# Evaluate classification performance
print(classification_report(y_test, y_pred))

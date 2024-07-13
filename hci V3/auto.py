import tkinter as tk
from tkinter import filedialog
import numpy as np
import joblib

def load_model():
    global svm_model
    model_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
    if model_path:
        svm_model = joblib.load(model_path)
        status_label.config(text="SVM model loaded successfully!")

def load_components():
    global ica_components, ica_components_reduced
    components_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
    if components_path:
        ica_components = joblib.load(components_path)
        n_components = 12
        ica_components_reduced = ica_components[:n_components]
        status_label.config(text="ICA components loaded and reduced successfully!")

def predict_label(sample_feature_vector):
    if svm_model is None or ica_components_reduced is None:
        return "Model or ICA components not loaded."

    try:
        if len(sample_feature_vector) != 22:
            raise ValueError("Invalid number of features. Expected 22 features.")

        # Apply ICA transformation and reduce dimensionality to 12 features
        sample_feature_vector_ica_reduced = np.dot(sample_feature_vector, ica_components_reduced.T)

        # Predict the label using the SVM model
        predicted_label = svm_model.predict(sample_feature_vector_ica_reduced.reshape(1, -1))
        predicted_label_class = label_to_class[predicted_label[0]]
        return predicted_label_class
    except Exception as e:
        return str(e)

def predict_for_all_samples():
    patients_signals = [
        [[-4.112318346, -5.39123962, -3.642817144, -2.570476399, -1.428649476, -2.298165701, -0.792005846, -0.352552721, -0.609839379, 0.488793433, 0.657813866, -0.440818947, -3.248436134, 4.209120957, 3.786569875, 2.348018193, 0.776128169, 1.012756774, 4.990370957, 2.819397399, 0.947026606, 2.851323481],
         [0.5, 1.2, 1.0, 0.8, 1.1, 0.7, -0.4, 0.6, 0.9, -0.3, 0.8, 0.5, -0.2, 0.4, -0.1, 0.3, 0.6, 0.8, -0.5, 0.4, 0.2, -0.6]],
        [[-1.1, -0.9, -1.2, -0.7, -1.3, -0.6, 0.4, -1.0, -0.8, 0.3, -1.4, -0.5, 0.2, -0.4, 0.1, -0.3, -0.6, -0.8, 0.5, -0.4, -0.2, 0.6],
         [2.1, 2.9, 3.2, 2.7, 3.3, 2.6, -0.4, 2.0, 2.8, -0.3, 3.4, 2.5, -0.2, 2.4, -0.1, 2.3, 2.6, 2.8, -0.5, 2.4, 2.2, -0.6]],
        [[0.8, 1.4, 1.6, 1.1, 1.7, 1.0, -0.4, 1.3, 1.5, -0.3, 1.8, 0.9, -0.2, 1.2, -0.1, 1.1, 1.4, 1.6, -0.5, 1.3, 1.1, -0.6],
         [-0.8, -1.4, -1.6, -1.1, -1.7, -1.0, 0.4, -1.3, -1.5, 0.3, -1.8, -0.9, 0.2, -1.2, 0.1, -1.1, -1.4, -1.6, 0.5, -1.3, -1.1, 0.6]],
        [[1.2, 2.1, 2.4, 1.8, 2.5, 1.7, -0.4, 2.0, 2.2, -0.3, 2.6, 1.6, -0.2, 1.9, -0.1, 1.8, 2.1, 2.3, -0.5, 2.0, 1.8, -0.6],
         [-1.2, -2.1, -2.4, -1.8, -2.5, -1.7, 0.4, -2.0, -2.2, 0.3, -2.6, -1.6, 0.2, -1.9, 0.1, -1.8, -2.1, -2.3, 0.5, -2.0, -1.8, 0.6]]
    ]

    results = []
    for i, patient_signals in enumerate(patients_signals):
        for j, signal in enumerate(patient_signals):
            predicted_label = predict_label(signal)
            results.append(f"Patient {i+1}, Signal {j+1}: {predicted_label}")

    result_text = "\n".join(results)
    result_label.config(text=result_text)

# Initialize GUI
root = tk.Tk()
root.title("SVM Prediction GUI")

# Load buttons
load_model_button = tk.Button(root, text="Load SVM Model", command=load_model)
load_model_button.pack()

load_components_button = tk.Button(root, text="Load ICA Components", command=load_components)
load_components_button.pack()

# Prediction button
predict_button = tk.Button(root, text="Predict Labels for All Samples", command=predict_for_all_samples)
predict_button.pack()

# Result label
result_label = tk.Label(root, text="", justify=tk.LEFT)
result_label.pack()

# Status label
status_label = tk.Label(root, text="")
status_label.pack()

# Label mapping
label_to_class = {0: 'sleep', 1: 'eat', 2: 'drink water', 3: 'go to bathroom'}

# Global variables
svm_model = None
ica_components = None
ica_components_reduced = None

root.mainloop()

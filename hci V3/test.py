import numpy as np
import joblib
import tkinter as tk
from tkinter import messagebox

def predict_label(sample_feature_vector):
    # Load the SVM model
    svm_model = joblib.load('svm_model.pkl')

    # Load the ICA components
    ica_components = joblib.load('ica_components.pkl')

    # Select top 12 ICA components with highest explained variance ratio
    n_components = 12
    ica_components_reduced = ica_components[:n_components]

    # Apply ICA transformation and reduce dimensionality to 12 features
    sample_feature_vector_ica_reduced = np.dot(sample_feature_vector, ica_components_reduced.T)

    # Predict the label using the SVM model
    predicted_label = svm_model.predict(sample_feature_vector_ica_reduced.reshape(1, -1))

    # Dictionary to map the predicted label to its corresponding activity
    label_to_class = {0: 'sleep', 1: 'eat', 2: 'drink water', 3: 'go to bathroom'}

    # Print debugging information
    print("Sample Feature Vector:", sample_feature_vector)
    print("ICA Reduced Feature Vector:", sample_feature_vector_ica_reduced)
    print("Predicted Label Index:", predicted_label[0])

    # Return the predicted label
    return label_to_class[predicted_label[0]]

def on_predict():
    sample_text = text_entry.get("1.0", 'end-1c')
    sample_lines = sample_text.split("\n")
    result_text.delete("1.0", tk.END)
    for line in sample_lines:
        if line.strip() == "":
            continue
        try:
            sample_feature_vector = np.array(list(map(float, line.split(","))))
            predicted_label = predict_label(sample_feature_vector)
            result_text.insert(tk.END, f"Predicted Label: {predicted_label}\n")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            return

# GUI
root = tk.Tk()
root.title("Signal Label Predictor")

# Input Text Entry
tk.Label(root, text="Enter Sample Signal (one per line, comma separated):").pack()
text_entry = tk.Text(root, height=10, width=50)
text_entry.pack()

# Predict Button
predict_button = tk.Button(root, text="Predict", command=on_predict)
predict_button.pack()

# Output Text
tk.Label(root, text="Prediction Results:").pack()
result_text = tk.Text(root, height=10, width=50)
result_text.pack()

root.mainloop()

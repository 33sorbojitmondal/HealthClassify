import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import sys
import json
from collections import Counter

def load_health_data(csv_file):
    health_data = []
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if all(row.values()):
                health_data.append({
                    'symptoms': row['symptoms'].split(';'),
                    'disease': row['disease'],
                    'medication': row['medication']
                })
    return health_data

def preprocess_data(health_data):
    if not health_data:
        raise ValueError("No valid data found in the CSV file.")
    
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform([item['symptoms'] for item in health_data])
    
    disease_encoder = LabelEncoder()
    medication_encoder = LabelEncoder()
    
    diseases = [item['disease'] for item in health_data]
    medications = [item['medication'] for item in health_data]
    
    # Filter out classes with only one sample
    disease_counts = Counter(diseases)
    medication_counts = Counter(medications)
    
    valid_diseases = [d for d in diseases if disease_counts[d] > 1]
    valid_medications = [m for m in medications if medication_counts[m] > 1]
    
    y_disease = disease_encoder.fit_transform(valid_diseases)
    y_medication = medication_encoder.fit_transform(valid_medications)
    
    # Update X to match the filtered y
    X_disease = X[:len(valid_diseases)]
    X_medication = X[:len(valid_medications)]
    
    return X_disease, X_medication, y_disease, y_medication, mlb, disease_encoder, medication_encoder

def print_class_distribution(y, name):
    unique, counts = np.unique(y, return_counts=True)
    print(f"{name} distribution:")
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c}")
    print()

def train_and_evaluate_models(X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print_class_distribution(y, f"{model_name} (Full dataset)")
    print_class_distribution(y_test, f"{model_name} (Test set)")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_test)
    
    return y_proba, y_test

def plot_roc_curve(y_proba, y_test, model_name):
    plt.figure(figsize=(10, 5))

    for i in range(y_proba.shape[1]):
        if np.sum(y_test == i) > 0:  # Only plot if class is present in test set
            fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} Prediction')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f'roc_curve_{model_name.lower()}.png')
    plt.close()

def main():
    csv_file = 'processed_health_data.csv'
    try:
        health_data = load_health_data(csv_file)
        X_disease, X_medication, y_disease, y_medication, mlb, disease_encoder, medication_encoder = preprocess_data(health_data)
        
        print("Dataset information:")
        print(f"Total samples: {len(health_data)}")
        print(f"Number of features: {X_disease.shape[1]}")
        print(f"Number of disease classes (after filtering): {len(np.unique(y_disease))}")
        print(f"Number of medication classes (after filtering): {len(np.unique(y_medication))}")
        print()
        
        disease_proba, y_disease_test = train_and_evaluate_models(X_disease, y_disease, "Disease")
        medication_proba, y_medication_test = train_and_evaluate_models(X_medication, y_medication, "Medication")
        
        plot_roc_curve(disease_proba, y_disease_test, "Disease")
        plot_roc_curve(medication_proba, y_medication_test, "Medication")
        
        result = {
            "message": "ROC curves and AUC statistics have been generated and saved as 'roc_curve_disease.png' and 'roc_curve_medication.png'.",
            "dataset_info": {
                "total_samples": len(health_data),
                "num_features": X_disease.shape[1],
                "num_disease_classes": len(np.unique(y_disease)),
                "num_medication_classes": len(np.unique(y_medication))
            }
        }
        
        print(json.dumps(result, indent=2))
    
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == '__main__':
    main()
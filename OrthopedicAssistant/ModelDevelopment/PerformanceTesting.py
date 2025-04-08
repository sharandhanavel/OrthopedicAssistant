import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load trained models
implant_model = joblib.load("../SavedModels/implant_model.pkl")
procedure_model = joblib.load("../SavedModels/procedure_model.pkl")

# Categories for encoding
categories = {
    "Gender": ["Male", "Female"],
    "Activity Level": ["Low", "Moderate", "High"],
    "Comorbidities": ["None", "Diabetes", "Rheumatoid Arthritis", "Osteoporosis", "Multiple"],
    "Smoking Status": ["Non-smoker", "Former Smoker", "Current Smoker"],
    "Alcohol Use": ["No", "Occasional", "Regular"],
    "Deformity": ["None", "Valgus", "Varus", "Rotational"],
    "Bone Quality": ["Normal", "Osteoporotic", "Severely Compromised"],
    "Scenario": ["Distal Femoral Fracture", "Proximal Tibial Fracture", "Patellar Fracture", "Osteoarthritis",
                 "Rheumatoid Arthritis", "Post-Traumatic Arthritis", "Primary Bone Tumor", "Metastatic Lesion",
                 "Mechanical Failure", "Patellofemoral Disorder", "Congenital Disorder", "Cartilage Injury",
                 "Ligamentous Injury", "Meniscal Damage"]
}

# Initialize LabelEncoder
label_encoder = LabelEncoder()


# Encode input function
def encode_input(input_value, category_list):
    if input_value not in category_list:
        raise ValueError(f"Invalid input: {input_value}. Expected one of: {category_list}")
    return label_encoder.fit(category_list).transform([input_value])[0]


# Generate synthetic test data
def generate_test_data(num_samples=300):
    np.random.seed(42)
    data = []
    for _ in range(num_samples):
        age = np.random.randint(15, 85)
        gender = np.random.choice(categories["Gender"])
        bmi = round(np.random.uniform(18.5, 40.0), 1)
        activity_level = np.random.choice(categories["Activity Level"])
        comorbidities = np.random.choice(categories["Comorbidities"])
        smoking_status = np.random.choice(categories["Smoking Status"])
        alcohol_use = np.random.choice(categories["Alcohol Use"])
        deformity = np.random.choice(categories["Deformity"])
        bone_quality = np.random.choice(categories["Bone Quality"])
        scenario = np.random.choice(categories["Scenario"])
        implant = np.random.choice(["Implant A", "Implant B", "Implant C"])  # Placeholder for true labels
        procedure = np.random.choice(["Procedure X", "Procedure Y", "Procedure Z"])  # Placeholder for true labels
        data.append([age, gender, bmi, activity_level, comorbidities, smoking_status, alcohol_use, deformity,
                     bone_quality, scenario, implant, procedure])
    return pd.DataFrame(data, columns=["Age", "Gender", "BMI", "Activity Level", "Comorbidities", "Smoking Status",
                                       "Alcohol Use", "Deformity", "Bone Quality", "Scenario", "Implant", "Procedure"])


# Evaluate model performance
def evaluate_model(data):
    # Encode features and extract true labels
    encoded_data = data.copy()
    for col, category_list in categories.items():
        encoded_data[col] = data[col].apply(lambda x: encode_input(x, category_list))

    X = encoded_data[["Age", "Gender", "BMI", "Activity Level", "Comorbidities",
                      "Smoking Status", "Alcohol Use", "Deformity", "Bone Quality", "Scenario"]]
    y_implant_true = data["Implant"]
    y_procedure_true = data["Procedure"]

    # Predictions
    y_implant_pred = implant_model.predict(X)
    y_procedure_pred = procedure_model.predict(X)

    # Performance metrics
    print("\n--- Implant Model Performance ---")
    print("Accuracy:", accuracy_score(y_implant_true, y_implant_pred))
    print("\nClassification Report:")
    print(classification_report(y_implant_true, y_implant_pred))

    print("\n--- Procedure Model Performance ---")
    print("Accuracy:", accuracy_score(y_procedure_true, y_procedure_pred))
    print("\nClassification Report:")
    print(classification_report(y_procedure_true, y_procedure_pred))

    # Confusion Matrices
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_implant_true, y_implant_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix: Implant Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix(y_procedure_true, y_procedure_pred), annot=True, fmt="d", cmap="Greens")
    plt.title("Confusion Matrix: Procedure Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()


# Main process
test_data = generate_test_data()
evaluate_model(test_data)

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained models
implant_model = joblib.load("../SavedModels/implant_model.pkl")
procedure_model = joblib.load("../SavedModels/procedure_model.pkl")

# Load LabelEncoder to encode inputs
label_encoder = LabelEncoder()

# Predefined list of categories for each input field to ensure correct input encoding
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


# Function to encode inputs based on LabelEncoder and category mapping
def encode_input(input_value, category_list):
    if input_value not in category_list:
        raise ValueError(f"Invalid input: {input_value}. Expected one of: {category_list}")
    return label_encoder.fit(category_list).transform([input_value])[0]


# Function to simulate the prediction process with predefined input values
def test_with_predefined_inputs():
    print("\n--- Knee Implant and Procedure Prediction ---\n")

    # Predefined input values (as an example)
    age = 65
    gender = "Female"
    bmi = 25.3
    activity_level = "Moderate"
    comorbidities = "Diabetes"
    smoking_status = "Non-smoker"
    alcohol_use = "Regular"
    deformity = "None"
    bone_quality = "Normal"
    scenario = "Mechanical Failure"

    print(
        f"Using the following predefined inputs:\nAge: {age}\nGender: {gender}\nBMI: {bmi}\nActivity Level: {activity_level}\n"
        f"Comorbidities: {comorbidities}\nSmoking Status: {smoking_status}\nAlcohol Use: {alcohol_use}\n"
        f"Deformity: {deformity}\nBone Quality: {bone_quality}\nScenario: {scenario}")

    try:
        # Encode user inputs
        encoded_inputs = [
            age,  # Age is a continuous feature, no need to encode it.
            encode_input(gender, categories["Gender"]),
            bmi,  # BMI is continuous, no need to encode it.
            encode_input(activity_level, categories["Activity Level"]),
            encode_input(comorbidities, categories["Comorbidities"]),
            encode_input(smoking_status, categories["Smoking Status"]),
            encode_input(alcohol_use, categories["Alcohol Use"]),
            encode_input(deformity, categories["Deformity"]),
            encode_input(bone_quality, categories["Bone Quality"]),
            encode_input(scenario, categories["Scenario"])
        ]

        # Prepare input as a DataFrame for the model
        input_data = pd.DataFrame([encoded_inputs], columns=["Age", "Gender", "BMI", "Activity Level", "Comorbidities",
                                                             "Smoking Status", "Alcohol Use", "Deformity",
                                                             "Bone Quality", "Scenario"])

        # Make predictions for Implant and Procedure
        implant_prediction = implant_model.predict(input_data)[0]
        procedure_prediction = procedure_model.predict(input_data)[0]

        # Output predictions in a structured format
        print("\n--- Prediction Results ---")
        print(f"Recommended Implant: {implant_prediction}")
        print(f"Recommended Procedure: {procedure_prediction}")

    except ValueError as e:
        print(f"Error: {e}")


# Call the function to start the prediction process with predefined inputs
test_with_predefined_inputs()

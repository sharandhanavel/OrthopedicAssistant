import joblib
import pandas as pd

implant_model_data = joblib.load("../SavedModels/implant_model_v3.pkl")
procedure_model_data = joblib.load("../SavedModels/procedure_model_v3.pkl")

implant_model = implant_model_data["model"]
procedure_model = procedure_model_data["model"]
feature_encoder = implant_model_data["feature_encoder"]
implant_encoder = implant_model_data["target_encoder"]
procedure_encoder = procedure_model_data["target_encoder"]

patient_input = {
    "Age": 25,
    "Gender": "Male",
    "BMI": 24.1,
    "ActivityLevel": "Athletic",
    "Comorbidities": "None",
    "Deformity": "None",
    "BoneQuality": "Normal",
    "Scenario": "Tumor"
}

input_df = pd.DataFrame([patient_input])
categorical_columns = ["Gender", "ActivityLevel", "Comorbidities", "Deformity", "BoneQuality", "Scenario"]
input_df[categorical_columns] = feature_encoder.transform(input_df[categorical_columns])

implant_prediction = implant_model.predict(input_df)[0]
procedure_prediction = procedure_model.predict(input_df)[0]

predicted_implant = implant_encoder.inverse_transform([implant_prediction])[0]
predicted_procedure = procedure_encoder.inverse_transform([procedure_prediction])[0]

print("🦴 Clinical Decision Support Prediction")
print("=======================================")
print(f"Patient Scenario         : {patient_input['Scenario']}")
print(f"Recommended Implant      : {predicted_implant}")
print(f"Recommended Procedure    : {predicted_procedure}")

try:
    implant_proba = implant_model.predict_proba(input_df)[0]
    procedure_proba = procedure_model.predict_proba(input_df)[0]
    print("\nPrediction Confidence:")
    print(f"Implant Confidence: {max(implant_proba):.2f}")
    print(f"Procedure Confidence: {max(procedure_proba):.2f}")
except AttributeError:
    print("\nNote: Model does not provide prediction probabilities.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the synthetic dataset
dataset_path = "../SavedDataset/RevisedDataSet.csv"  # Path to the dataset file
df = pd.read_csv(dataset_path)


# Preprocessing
def preprocess_data(df):
    # Encode categorical variables
    label_encoders = {}
    for col in ["Gender", "Activity Level", "Comorbidities", "Smoking Status",
                "Alcohol Use", "Deformity", "Bone Quality", "Scenario"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Encode target variables
    implant_le = LabelEncoder()
    df["Recommended Implant Encoded"] = implant_le.fit_transform(df["Recommended Implant"])

    procedure_le = LabelEncoder()
    df["Recommended Procedure Encoded"] = procedure_le.fit_transform(df["Recommended Procedure"])

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ["Age", "BMI"]
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df, implant_le, procedure_le, scaler


# Preprocess the data
df_processed, implant_le, procedure_le, scaler = preprocess_data(df)

# Features for implant prediction
feature_cols = ["Age", "Gender", "BMI", "Activity Level", "Comorbidities",
                "Smoking Status", "Alcohol Use", "Deformity", "Bone Quality", "Scenario"]

X = df_processed[feature_cols]
y_implant = df_processed["Recommended Implant Encoded"]

# Train-test split for implant prediction
X_train_implant, X_test_implant, y_train_implant, y_test_implant = train_test_split(
    X, y_implant, test_size=0.2, random_state=42, stratify=y_implant
)

# Define and train the implant prediction model
implant_model = RandomForestClassifier(n_estimators=200, random_state=42)
implant_model.fit(X_train_implant, y_train_implant)

# Evaluate implant model
implant_train_score = implant_model.score(X_train_implant, y_train_implant)
implant_test_score = implant_model.score(X_test_implant, y_test_implant)
print(f"Implant Model - Training Accuracy: {implant_train_score:.2f}")
print(f"Implant Model - Test Accuracy: {implant_test_score:.2f}")

# Save the implant model and encoders
joblib.dump(implant_model, "../SavedModels/ImplantModel.pkl")
joblib.dump(implant_le, "../SavedModels/ImplantLabelEncoders.pkl")
joblib.dump(scaler, "../SavedModels/ImplantScalar.pkl")
print("Implant model and encoders saved successfully.")

# Prepare data for procedure prediction
# We'll use the entire dataset, adding the encoded implant as a feature
X_procedure = X.copy()
X_procedure["Recommended Implant Encoded"] = df_processed["Recommended Implant Encoded"]

y_procedure = df_processed["Recommended Procedure Encoded"]

# Train-test split for procedure prediction
X_train_proc, X_test_proc, y_train_proc, y_test_proc = train_test_split(
    X_procedure, y_procedure, test_size=0.2, random_state=42, stratify=y_procedure
)

# Define and train the procedure prediction model
procedure_model = RandomForestClassifier(n_estimators=200, random_state=42)
procedure_model.fit(X_train_proc, y_train_proc)

# Evaluate procedure model
procedure_train_score = procedure_model.score(X_train_proc, y_train_proc)
procedure_test_score = procedure_model.score(X_test_proc, y_test_proc)
print(f"Procedure Model - Training Accuracy: {procedure_train_score:.2f}")
print(f"Procedure Model - Test Accuracy: {procedure_test_score:.2f}")

# Save the procedure model and encoders
joblib.dump(procedure_model, "../SavedModels/ProcedureModel.pkl")
joblib.dump(procedure_le, "../SavedModels/ProcedureLabelEncoder.pkl")
print("Procedure model and encoders saved successfully.")

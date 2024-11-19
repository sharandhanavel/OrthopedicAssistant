import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import os
import joblib

# Load the synthetic dataset
dataset_path = "../SavedDataset/SyntheticDataSet.csv"  # Path to the dataset file
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

    # Separate features and targets
    X = df.drop(["Recommended Implant", "Recommended Procedure"], axis=1)
    y_implant = LabelEncoder().fit_transform(df["Recommended Implant"])
    y_procedure = LabelEncoder().fit_transform(df["Recommended Procedure"])

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_implant, y_procedure, label_encoders, scaler


# Preprocess the data
X, y_implant, y_procedure, label_encoders, scaler = preprocess_data(df)

# Combine targets for multi-output classification
y = pd.DataFrame({"Implant": y_implant, "Procedure": y_procedure})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
multi_output_model = MultiOutputClassifier(rf_model)

# Train the model
multi_output_model.fit(X_train, y_train)

# Evaluate the model
train_score = multi_output_model.score(X_train, y_train)
test_score = multi_output_model.score(X_test, y_test)
print(f"Training Accuracy: {train_score:.2f}")
print(f"Test Accuracy: {test_score:.2f}")

# Save the model and preprocessing objects
joblib.dump(multi_output_model, "../SavedModels/DecisionMakingModel.pkl")
joblib.dump(label_encoders, "../SavedModels/LabelEncoders.pkl")
joblib.dump(scaler, "../SavedModels/Scaler.pkl")

print("Model and preprocessing objects saved successfully.")

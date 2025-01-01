import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the synthetic dataset
data = pd.read_csv("../SavedDataset/OptimizedDataSet.csv")

# Preprocessing
# Encode categorical features using LabelEncoder
label_encoder = LabelEncoder()

# Columns to encode
categorical_columns = ["Gender", "Activity Level", "Comorbidities", "Smoking Status", "Alcohol Use", "Deformity", "Bone Quality", "Scenario"]

# Apply LabelEncoder to each categorical column
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Features and target for Implant model
X = data.drop(columns=["Recommended Implant", "Recommended Procedure"])
y_implant = data["Recommended Implant"]

# Features and target for Procedure model
y_procedure = data["Recommended Procedure"]

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train_implant, y_test_implant, y_train_procedure, y_test_procedure = train_test_split(
    X, y_implant, y_procedure, test_size=0.2, random_state=42
)

# Initialize RandomForestClassifier for both models
implant_model = RandomForestClassifier(n_estimators=100, random_state=42)
procedure_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Implant model
implant_model.fit(X_train, y_train_implant)

# Train the Procedure model
procedure_model.fit(X_train, y_train_procedure)

# Predictions on the test set
implant_predictions = implant_model.predict(X_test)
procedure_predictions = procedure_model.predict(X_test)

# Evaluate the Implant model
implant_accuracy = accuracy_score(y_test_implant, implant_predictions)
implant_report = classification_report(y_test_implant, implant_predictions, zero_division=1)

# Evaluate the Procedure model
procedure_accuracy = accuracy_score(y_test_procedure, procedure_predictions)
procedure_report = classification_report(y_test_procedure, procedure_predictions, zero_division=1)

# Print evaluation results
print(f"Implant Model Accuracy: {implant_accuracy:.4f}")
print(f"Implant Model Classification Report:\n{implant_report}")
print(f"Procedure Model Accuracy: {procedure_accuracy:.4f}")
print(f"Procedure Model Classification Report:\n{procedure_report}")

# Save the trained models to disk
joblib.dump(implant_model, "../SavedModels/implant_model.pkl")
joblib.dump(procedure_model, "../SavedModels/procedure_model.pkl")

print("Models saved successfully.")

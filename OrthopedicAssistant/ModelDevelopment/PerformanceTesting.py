from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from PredictionModel import y_test_implant, X_test_implant, X_test_proc, \
    y_test_proc

# Load saved models and encoders
implant_model = joblib.load("../SavedModels/ImplantModel.pkl")
implant_le = joblib.load("../SavedModels/ImplantLabelEncoders.pkl")
procedure_model = joblib.load("../SavedModels/ProcedureModel.pkl")
procedure_le = joblib.load("../SavedModels/ProcedureLabelEncoder.pkl")
scaler = joblib.load("../SavedModels/ImplantScalar.pkl")

# Predict on test data
y_pred_implant = implant_model.predict(X_test_implant)
y_pred_procedure = procedure_model.predict(X_test_proc)

# Decode predictions
y_test_implant_decoded = implant_le.inverse_transform(y_test_implant)
y_pred_implant_decoded = implant_le.inverse_transform(y_pred_implant)

y_test_proc_decoded = procedure_le.inverse_transform(y_test_proc)
y_pred_proc_decoded = procedure_le.inverse_transform(y_pred_procedure)

# Implant Model Evaluation
print("Implant Model Classification Report:")
print(classification_report(y_test_implant_decoded, y_pred_implant_decoded))

# Plot Confusion Matrix for Implant Model
cm_implant = confusion_matrix(y_test_implant_decoded, y_pred_implant_decoded)
plt.figure(figsize=(10,8))
sns.heatmap(cm_implant, annot=True, fmt='d', cmap='Blues',
            xticklabels=implant_le.classes_,
            yticklabels=implant_le.classes_)
plt.title("Implant Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Procedure Model Evaluation
print("Procedure Model Classification Report:")
print(classification_report(y_test_proc_decoded, y_pred_proc_decoded))

# Plot Confusion Matrix for Procedure Model
cm_procedure = confusion_matrix(y_test_proc_decoded, y_pred_proc_decoded)
plt.figure(figsize=(12,10))
sns.heatmap(cm_procedure, annot=True, fmt='d', cmap='Greens',
            xticklabels=procedure_le.classes_,
            yticklabels=procedure_le.classes_)
plt.title("Procedure Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

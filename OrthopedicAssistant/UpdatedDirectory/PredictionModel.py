import pandas as pd
import joblib
import logging
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Loading dataset...")
data = pd.read_csv("../SavedDataset/ClinicallyValid_KneeImplant_Dataset.csv")

categorical_columns = ["Gender", "ActivityLevel", "Comorbidities", "Deformity", "BoneQuality", "Scenario"]

feature_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
data[categorical_columns] = feature_encoder.fit_transform(data[categorical_columns])

implant_encoder = LabelEncoder()
procedure_encoder = LabelEncoder()

data["RecommendedImplant"] = implant_encoder.fit_transform(data["RecommendedImplant"])
data["RecommendedProcedure"] = procedure_encoder.fit_transform(data["RecommendedProcedure"])

X = data.drop(columns=["RecommendedImplant", "RecommendedProcedure"])
y_implant = data["RecommendedImplant"]
y_procedure = data["RecommendedProcedure"]

X_train, X_test, y_train_implant, y_test_implant, y_train_procedure, y_test_procedure = train_test_split(
    X, y_implant, y_procedure, test_size=0.2, stratify=y_implant, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote_implant, y_train_smote_implant = smote.fit_resample(X_train, y_train_implant)
X_train_smote_procedure, y_train_smote_procedure = smote.fit_resample(X_train, y_train_procedure)

def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selection", SelectFromModel(XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42), threshold="median")),
        ("clf", XGBClassifier(eval_metric='mlogloss', random_state=42))
    ])

param_dist = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [4, 6, 10],
    'clf__learning_rate': [0.01, 0.1, 0.3],
    'clf__subsample': [0.7, 1.0],
    'clf__colsample_bytree': [0.7, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

logging.info("Training implant model...")
implant_model = RandomizedSearchCV(build_pipeline(), param_distributions=param_dist, cv=cv,
                                    n_iter=10, scoring='accuracy', n_jobs=-1, random_state=42)
implant_model.fit(X_train_smote_implant, y_train_smote_implant)

logging.info("Training procedure model...")
procedure_model = RandomizedSearchCV(build_pipeline(), param_distributions=param_dist, cv=cv,
                                      n_iter=10, scoring='accuracy', n_jobs=-1, random_state=42)
procedure_model.fit(X_train_smote_procedure, y_train_smote_procedure)

implant_preds = implant_model.predict(X_test)
procedure_preds = procedure_model.predict(X_test)

logging.info(f"Implant Model Accuracy: {accuracy_score(y_test_implant, implant_preds):.4f}")
logging.info(f"Procedure Model Accuracy: {accuracy_score(y_test_procedure, procedure_preds):.4f}")

logging.info("Implant Classification Report:\n" + classification_report(y_test_implant, implant_preds, zero_division=1))
logging.info("Procedure Classification Report:\n" + classification_report(y_test_procedure, procedure_preds, zero_division=1))

ConfusionMatrixDisplay(confusion_matrix(y_test_implant, implant_preds)).plot()
plt.title("Implant Prediction Confusion Matrix")
plt.show()

ConfusionMatrixDisplay(confusion_matrix(y_test_procedure, procedure_preds)).plot()
plt.title("Procedure Prediction Confusion Matrix")
plt.show()

explainer = shap.Explainer(implant_model.best_estimator_.named_steps['clf'])
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

logging.info("Saving models...")
joblib.dump({
    "model": implant_model.best_estimator_,
    "version": "3.1",
    "feature_encoder": feature_encoder,
    "target_encoder": implant_encoder
}, "../SavedModels/implant_model_v3.pkl")

joblib.dump({
    "model": procedure_model.best_estimator_,
    "version": "3.1",
    "feature_encoder": feature_encoder,
    "target_encoder": procedure_encoder
}, "../SavedModels/procedure_model_v3.pkl")

logging.info("Clinical decision support models with SHAP explainability saved successfully.")

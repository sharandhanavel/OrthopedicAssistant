import pandas as pd
import numpy as np

np.random.seed(42)

mappings = {
    "implants": {
        "Fractures": {
            "Distal Femoral Replacement": {"criteria": ["Open fracture", "Comminuted fracture"], "procedure": "ORIF with Bone Grafting"},
            "Tibial Plateau Prosthesis": {"criteria": ["Schatzker V-VI", "Bone loss >5mm"], "procedure": "ORIF with Augmentation"}
        },
        "Arthritis": {
            "Unicompartmental Knee Replacement": {"criteria": ["Isolated medial/lateral OA", "Intact ligaments", "BMI<35"], "procedure": "Minimally Invasive UKA"},
            "Total Knee Replacement": {"criteria": ["Bicompartmental/tricompartmental OA", "Deformity >15°"], "procedure": "Standard TKA with Measured Resection"}
        },
        "Tumors": {
            "Custom Tumor Prosthesis": {"criteria": ["Primary bone tumors", "Metastatic lesions with >50% bone loss"], "procedure": "Wide Resection with Reconstruction"}
        },
        "Instability": {
            "Hinged Knee Replacement": {"criteria": ["Collateral ligament insufficiency", "Severe bone loss"], "procedure": "Revision TKA with Constrained Implant"},
            "Patellofemoral Replacement": {"criteria": ["Isolated patellofemoral arthritis", "Normal tibiofemoral joints"], "procedure": "Patellofemoral Arthroplasty"}
        }
    },
    "procedure_sequences": {
        "Primary TKA": ["Medial parapatellar approach", "Gap balancing", "Cruciate-retaining/sacrificing"],
        "Revision TKA": ["Extended approach", "Debridement", "Stemmed components"],
        "Trauma": ["Fracture reduction", "Joint line restoration", "Augmentation"]
    }
}

def generate_synthetic_data(num_samples=10000):
    scenarios = {
        "Primary Osteoarthritis": 0.55,
        "Post-Traumatic Arthritis": 0.18,
        "Inflammatory Arthritis": 0.12,
        "Periprosthetic Fracture": 0.05,
        "Aseptic Loosening": 0.04,
        "Prosthetic Joint Infection": 0.03,
        "Osteonecrosis": 0.02,
        "Tumor": 0.01
    }

    scenario_names = list(scenarios.keys())
    scenario_probs = np.array(list(scenarios.values()))
    scenario_probs /= scenario_probs.sum()

    data = []
    for _ in range(num_samples):
        age = int(np.clip(np.random.normal(68, 12), 40, 90))
        gender = np.random.choice(["Male", "Female"], p=[0.45, 0.55])
        bmi = round(np.clip(np.random.normal(32, 6), 18, 45), 1)
        activity_level = np.random.choice(["Sedentary", "Household", "Community", "Athletic"])
        comorbidities = np.random.choice(["None", "Diabetes", "Cardiovascular", "Osteoporosis", "Rheumatoid"], p=[0.6, 0.15, 0.15, 0.05, 0.05])
        deformity = np.random.choice(["None", "Varus <10°", "Varus 10-20°", "Valgus <10°", "Valgus 10-20°"], p=[0.3, 0.4, 0.15, 0.1, 0.05])
        bone_quality = np.random.choice(["Normal", "Osteopenic", "Osteoporotic"], p=[0.6, 0.3, 0.1])
        scenario = np.random.choice(scenario_names, p=scenario_probs)

        implant, procedure = select_implant_procedure(scenario, age, bmi, deformity, comorbidities)

        data.append({
            "Age": age,
            "Gender": gender,
            "BMI": bmi,
            "ActivityLevel": activity_level,
            "Comorbidities": comorbidities,
            "Deformity": deformity,
            "BoneQuality": bone_quality,
            "Scenario": scenario,
            "RecommendedImplant": implant,
            "RecommendedProcedure": procedure
        })

    return pd.DataFrame(data)

def select_implant_procedure(scenario, age, bmi, deformity, comorbidities):
    if "Osteoarthritis" in scenario:
        if "Varus <10°" in deformity and bmi < 35 and comorbidities == "None":
            return ("Unicompartmental Knee Replacement", "Minimally Invasive UKA")
        else:
            return ("Total Knee Replacement", "Standard TKA with Measured Resection")
    elif "Post-Traumatic" in scenario:
        if age < 60 and "Varus" in deformity:
            return ("Total Knee Replacement", "Post-Traumatic TKA with Augmentation")
        else:
            return ("Total Knee Replacement", "Constrained Condylar Knee")
    elif "Inflammatory" in scenario:
        return ("Total Knee Replacement", "CCK with Stem Extension")
    elif "Fracture" in scenario:
        return ("Distal Femoral Replacement", "ORIF with Bone Grafting")
    elif "Infection" in scenario:
        return ("Spacer", "Two-Stage Revision")
    elif "Tumor" in scenario:
        return ("Custom Tumor Prosthesis", "Wide Resection with Reconstruction")
    return ("Total Knee Replacement", "Standard TKA with Measured Resection")

synthetic_data = generate_synthetic_data(10000)
file_path = "../SavedDataset/ClinicallyValid_KneeImplant_Dataset.csv"
synthetic_data.to_csv(file_path, index=False)

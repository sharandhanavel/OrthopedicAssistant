import pandas as pd
import random
import numpy as np

# Seed for reproducibility
random.seed(42)
np.random.seed(42)


# Define possible values for each aspect
def generate_synthetic_data(num_samples=1000):
    # Scenarios
    scenarios = [
        "Distal Femoral Fracture", "Proximal Tibial Fracture", "Patellar Fracture",
        "Complex Periarticular Fracture", "Ligamentous Injury", "Meniscal Damage",
        "Cartilage Injury", "Post-Traumatic Arthritis", "Osteoarthritis",
        "Rheumatoid Arthritis", "Avascular Necrosis", "Crystal Arthropathy",
        "Osteochondritis Dissecans", "Juvenile Idiopathic Arthritis", "Physeal Injury",
        "Septic Arthritis", "Primary Bone Tumor", "Metastatic Lesion",
        "Mechanical Failure", "Patellofemoral Disorder", "Congenital Disorder"
    ]

    # Implants
    implants = [
        "Unicompartmental Knee Replacement", "Total Knee Replacement",
        "Distal Femoral Replacement", "Tibial Plateau Prosthesis",
        "Hinged Knee Replacement", "Patellofemoral Joint Replacement",
        "Osteochondral Allograft", "Custom Tumor Prosthesis"
    ]

    # Procedures
    procedures = [
        "ORIF (Open Reduction Internal Fixation)", "Arthroscopic Meniscal Repair",
        "Autologous Chondrocyte Implantation", "Two-Stage Revision with Spacer",
        "Joint Resurfacing", "Total Knee Arthroplasty", "Partial Knee Arthroplasty",
        "Bone Grafting", "Ligament Reconstruction"
    ]

    # Generate synthetic patient characteristics
    data = []
    for _ in range(num_samples):
        # Patient characteristics
        age = random.randint(15, 85)
        gender = random.choice(["Male", "Female"])
        bmi = round(random.uniform(18.5, 40.0), 1)
        activity_level = random.choice(["Low", "Moderate", "High"])
        comorbidities = random.choice(["None", "Diabetes", "Rheumatoid Arthritis", "Osteoporosis", "Multiple"])
        smoking_status = random.choice(["Non-smoker", "Former Smoker", "Current Smoker"])
        alcohol_use = random.choice(["No", "Occasional", "Regular"])
        deformity = random.choice(["None", "Valgus", "Varus", "Rotational"])
        bone_quality = random.choice(["Normal", "Osteoporotic", "Severely Compromised"])

        # Injury/condition scenario
        scenario = random.choice(scenarios)

        # Choose implant and procedure based on scenario
        if scenario in ["Distal Femoral Fracture", "Proximal Tibial Fracture", "Patellar Fracture"]:
            implant = random.choice(["Distal Femoral Replacement", "Tibial Plateau Prosthesis"])
            procedure = random.choice(["ORIF (Open Reduction Internal Fixation)", "Bone Grafting"])
        elif scenario in ["Osteoarthritis", "Rheumatoid Arthritis", "Post-Traumatic Arthritis"]:
            implant = random.choice(["Unicompartmental Knee Replacement", "Total Knee Replacement"])
            procedure = random.choice(["Total Knee Arthroplasty", "Partial Knee Arthroplasty"])
        elif scenario in ["Meniscal Damage", "Cartilage Injury"]:
            implant = random.choice(["Osteochondral Allograft", "Unicompartmental Knee Replacement"])
            procedure = random.choice(["Arthroscopic Meniscal Repair", "Autologous Chondrocyte Implantation"])
        elif scenario in ["Septic Arthritis", "Osteomyelitis"]:
            implant = random.choice(["Total Knee Replacement", "Hinged Knee Replacement"])
            procedure = "Two-Stage Revision with Spacer"
        elif scenario in ["Primary Bone Tumor", "Metastatic Lesion"]:
            implant = "Custom Tumor Prosthesis"
            procedure = "Bone Grafting"
        else:
            implant = random.choice(implants)
            procedure = random.choice(procedures)

        # Append to data
        data.append({
            "Age": age,
            "Gender": gender,
            "BMI": bmi,
            "Activity Level": activity_level,
            "Comorbidities": comorbidities,
            "Smoking Status": smoking_status,
            "Alcohol Use": alcohol_use,
            "Deformity": deformity,
            "Bone Quality": bone_quality,
            "Scenario": scenario,
            "Recommended Implant": implant,
            "Recommended Procedure": procedure
        })

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df


# Generate dataset
synthetic_data = generate_synthetic_data(1000)

# Save dataset to CSV
synthetic_data.to_csv("../SavedDataset/SyntheticDataSet.csv", index=False)
print("Synthetic dataset generated and saved as 'SyntheticDataSet.csv'.")

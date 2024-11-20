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

    # Procedures mapped to implants
    implant_procedure_map = {
        "Unicompartmental Knee Replacement": ["Partial Knee Arthroplasty", "Total Knee Arthroplasty"],
        "Total Knee Replacement": ["Total Knee Arthroplasty"],
        "Distal Femoral Replacement": ["ORIF (Open Reduction Internal Fixation)", "Bone Grafting"],
        "Tibial Plateau Prosthesis": ["ORIF (Open Reduction Internal Fixation)", "Bone Grafting"],
        "Hinged Knee Replacement": ["Two-Stage Revision with Spacer", "Total Knee Arthroplasty"],
        "Patellofemoral Joint Replacement": ["Joint Resurfacing", "Patellar Resurfacing"],
        "Osteochondral Allograft": ["Autologous Chondrocyte Implantation", "Arthroscopic Meniscal Repair"],
        "Custom Tumor Prosthesis": ["Bone Grafting", "Wide Tumor Excision"]
    }

    # Procedures not directly tied to implants (for other scenarios)
    other_procedures = [
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

        # Choose implant and corresponding procedure based on scenario
        if scenario in ["Distal Femoral Fracture", "Proximal Tibial Fracture", "Patellar Fracture"]:
            implant = random.choice(["Distal Femoral Replacement", "Tibial Plateau Prosthesis"])
        elif scenario in ["Osteoarthritis", "Rheumatoid Arthritis", "Post-Traumatic Arthritis"]:
            implant = random.choice(["Unicompartmental Knee Replacement", "Total Knee Replacement"])
        elif scenario in ["Meniscal Damage", "Cartilage Injury"]:
            implant = random.choice(["Osteochondral Allograft", "Unicompartmental Knee Replacement"])
        elif scenario in ["Septic Arthritis", "Osteomyelitis"]:
            implant = random.choice(["Total Knee Replacement", "Hinged Knee Replacement"])
        elif scenario in ["Primary Bone Tumor", "Metastatic Lesion"]:
            implant = "Custom Tumor Prosthesis"
        elif scenario in ["Mechanical Failure", "Patellofemoral Disorder", "Congenital Disorder"]:
            implant = random.choice(
                ["Hinged Knee Replacement", "Patellofemoral Joint Replacement", "Total Knee Replacement"])
        else:
            implant = random.choice(implants)

        # Select procedure based on implant
        if implant in implant_procedure_map:
            procedure = random.choice(implant_procedure_map[implant])
        else:
            procedure = random.choice(other_procedures)

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
synthetic_data.to_csv("../SavedDataSet/RevisedDataSet.csv", index=False)
print("Synthetic dataset generated and saved as 'RevisedDataSet.csv'.")

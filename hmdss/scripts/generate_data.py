import os
import json
import pandas as pd
import numpy as np
from fpdf import FPDF
from datetime import datetime, timedelta

DATA_DIR = "hmdss/data"
POLICIES_DIR = os.path.join(DATA_DIR, "policies")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
EHR_DIR = os.path.join(DATA_DIR, "ehr")

def ensure_dirs():
    os.makedirs(POLICIES_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(EHR_DIR, exist_ok=True)

def generate_pdf_policy():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    title = "Surgical Safety and Staffing Protocols"
    content = """
    1. Introduction
    This document outlines the standard operating procedures for surgical staffing and resource allocation at City General Hospital.

    2. Surgical Staffing Requirements
    - Major Surgeries (e.g., Cardiac, Neuro): Minimum of 2 lead surgeons, 1 anesthesiologist, 3 OR nurses.
    - Minor Surgeries (e.g., Appendectomy): 1 lead surgeon, 1 anesthesiologist, 2 OR nurses.
    - Emergency Procedures: Rapid response team must be available within 10 minutes.

    3. Resource Allocation
    - Ventilators must be reserved for critical care units.
    - PPE stock levels must be checked every shift.
    - Surgical kits must be sterilized and prepared 2 hours before scheduled procedures.

    4. Scheduling Guidelines
    - Elective surgeries should be scheduled between 08:00 and 16:00.
    - A 30-minute buffer is required between procedures for cleaning and prep.
    - Staff shifts are 12 hours max, with a mandatory 10-hour rest period.
    """

    pdf.cell(200, 10, txt=title, ln=1, align='C')
    pdf.multi_cell(0, 10, txt=content)

    filepath = os.path.join(POLICIES_DIR, "surgical_policy.pdf")
    pdf.output(filepath)
    print(f"Generated PDF policy at {filepath}")

def generate_csv_logs():
    # Generate 1000 records of historical data
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=365)
    dates = [start_date + timedelta(days=i) for i in range(365)]

    data = []
    for date in dates:
        # Simulate data: Day of week affects volume
        is_weekend = date.weekday() >= 5
        base_volume = 50 if is_weekend else 120
        volume = int(np.random.normal(base_volume, 15))
        volume = max(0, volume)

        staff_available = int(volume * 0.15 + np.random.normal(5, 2))
        surgeries_completed = int(volume * 0.4 + np.random.normal(2, 1))

        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "patient_volume": volume,
            "staff_on_duty": staff_available,
            "surgeries_completed": surgeries_completed,
            "er_wait_time_minutes": int(volume * 0.5 + np.random.normal(10, 5))
        })

    df = pd.DataFrame(data)
    filepath = os.path.join(LOGS_DIR, "resource_logs.csv")
    df.to_csv(filepath, index=False)
    print(f"Generated CSV logs at {filepath}")

def generate_json_ehr():
    metadata = [
        {"patient_id": "P001", "department": "Cardiology", "status": "Critical", "last_visit": "2023-10-01"},
        {"patient_id": "P002", "department": "Orthopedics", "status": "Stable", "last_visit": "2023-10-05"},
        {"patient_id": "P003", "department": "Neurology", "status": "Observation", "last_visit": "2023-09-20"},
        # Add more dummy data
    ]

    filepath = os.path.join(EHR_DIR, "metadata.json")
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Generated JSON EHR metadata at {filepath}")

if __name__ == "__main__":
    ensure_dirs()
    generate_pdf_policy()
    generate_csv_logs()
    generate_json_ehr()

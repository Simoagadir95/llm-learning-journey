# 01b_generate_larger_dataset.py
"""
Génération d'un dataset de 500 exemples avec variations
"""

import json
import random
import os

print("Génération d'un dataset étendu (500 exemples)...")

# Templates de base par catégorie
templates = {
    "MACHINE_BREAKDOWN": [
        "{machine} stopped unexpectedly with error {error}. {symptom}.",
        "{machine} experienced {component} failure. Production halted on {line}.",
        "{system} malfunction on {machine}. {action} activated.",
        "{component} failure on {machine}. Error code {error}.",
        "{machine} {issue} detected. {symptom}.",
    ],
    
    "QUALITY_ISSUE": [
        "Quality inspection failed on Batch {batch}. {percent}% of units {defect}.",
        "{defect_type} detected. Parts measuring {measurement} {direction} specification.",
        "{defect} found on {count} units. {description}.",
        "{test} failure on {count} assemblies. {method} inspection failed.",
        "{property} {issue}. {detail}.",
    ],
    
    "SAFETY_ISSUE": [
        "Operator reported {equipment} {issue} on {machine}.",
        "{hazard} detected. {description}.",
        "{violation} observed during {activity}.",
        "{incident} in {location}. {outcome}.",
        "{equipment} {condition}. {severity}.",
    ],
    
    "MAINTENANCE": [
        "{activity} completed on {machine}. {status}.",
        "{task} performed on {equipment}. {result}.",
        "{inspection} on {system}. {finding}.",
        "{maintenance_type} on {equipment}. {outcome}.",
        "{procedure} completed. {verification}.",
    ],
    
    "SUPPLY_CHAIN": [
        "{material} {issue} detected. {detail}.",
        "Supplier {problem}. {impact}.",
        "{issue_type} with {item}. {action}.",
        "{incident} on {shipment}. {quantity} affected.",
        "{change} received. {detail}.",
    ]
}

# Variables pour générer des variations
variables = {
    "machine": ["CNC-A3", "CNC-B5", "CNC-C2", "Press #3", "Press #4", "Robot #7", 
                "Conveyor Line 2", "Assembly Line 3", "Furnace #2", "Machine #5"],
    "error": ["E402", "E501", "E303", "E404", "E505", "E601"],
    "symptom": ["Hydraulic pressure low", "Temperature sensor offline", "Communication lost",
                "Spindle overheating", "Encoder failure", "Power fluctuation"],
    "component": ["servo motor", "hydraulic pump", "PLC controller", "bearing", 
                  "spindle", "encoder", "actuator"],
    "line": ["Line 1", "Line 2", "Line 3", "Assembly Area", "Production Floor"],
    "system": ["Conveyor belt", "Cooling system", "Hydraulic system", "Pneumatic system"],
    "action": ["Emergency stop", "Safety shutdown", "Alarm", "Lockout"],
    "issue": ["hydraulic leak", "overheating", "vibration", "noise", "pressure drop"],
    
    "batch": ["B2045", "B3021", "A302", "C405", "B4012"],
    "percent": ["15", "20", "12", "8", "25"],
    "defect": ["outside tolerance", "with surface defects", "below specification"],
    "defect_type": ["Dimensional variance", "Surface finish defects", "Color mismatch"],
    "measurement": ["0.3mm", "0.5mm", "0.2mm", "1mm"],
    "direction": ["over", "under", "outside"],
    "count": ["8", "12", "15", "5", "23"],
    "description": ["Rough texture", "Scratches visible", "Contamination present"],
    "test": ["Weld penetration", "Leak test", "Hardness test", "Functional test"],
    "method": ["X-ray", "Visual", "Ultrasonic"],
    "property": ["Paint thickness", "Torque values", "Thread depth"],
    
    "equipment": ["safety guard", "emergency stop", "light curtain", "machine guarding"],
    "hazard": ["Chemical spill", "Slip hazard", "Fall hazard", "Pinch point"],
    "violation": ["LOTO violation", "PPE non-compliance", "Aisle blockage"],
    "incident": ["Near-miss", "Forklift incident", "Equipment contact"],
    "location": ["Warehouse Zone B", "Assembly Area", "Mixing station"],
    "outcome": ["No injuries", "Minor injury", "Medical attention required"],
    "condition": ["door open", "bypass detected", "missing", "malfunction"],
    "severity": ["Immediate inspection required", "Safety concern", "Critical issue"],
    
    "activity": ["Preventive maintenance", "Calibration", "Filter replacement"],
    "status": ["All systems operational", "Ready for production", "Testing complete"],
    "task": ["Lubrication", "Belt adjustment", "Sensor cleaning"],
    "result": ["Performance improved", "Within specification", "Completed successfully"],
    "inspection": ["Vibration analysis", "Thermography scan", "Visual inspection"],
    "finding": ["No issues detected", "Baseline updated", "Normal operation"],
    "maintenance_type": ["Bearing replacement", "Software update", "Alignment check"],
    "procedure": ["Safety interlock test", "Emergency lighting test"],
    "verification": ["Documented in CMMS", "Signed off", "Certified"],
    
    "material": ["Steel stock", "Raw materials", "Packaging materials"],
    "problem": ["delivery delay", "quality issue", "capacity constraints"],
    "impact": ["Production delayed 2 hours", "Expected 3 days late", "Shortage detected"],
    "issue_type": ["Wrong part number", "Freight damage", "Inventory discrepancy"],
    "item": ["incoming components", "shipment", "materials"],
    "shipment": ["incoming shipment", "delivery PO-2045"],
    "quantity": ["15 pallets", "50 units", "8 units"],
    "change": ["Supplier notification", "Lead time increase", "Price increase"],
}

def generate_from_template(template, category):
    """Génère un exemple en remplissant le template avec des variables aléatoires"""
    text = template
    # Remplacer les {placeholders} par des valeurs aléatoires
    import re
    placeholders = re.findall(r'\{(\w+)\}', template)
    for placeholder in placeholders:
        if placeholder in variables:
            value = random.choice(variables[placeholder])
            text = text.replace(f"{{{placeholder}}}", value, 1)
    return text

# Générer le dataset
dataset = []
examples_per_category = 100

for category, template_list in templates.items():
    print(f"Génération de {examples_per_category} exemples pour {category}...")
    for i in range(examples_per_category):
        template = random.choice(template_list)
        text = generate_from_template(template, category)
        dataset.append({
            "text": text,
            "label": category
        })

# Mélanger
random.shuffle(dataset)

# Split 70/15/15
n = len(dataset)
train_size = int(0.7 * n)
val_size = int(0.15 * n)

train_data = dataset[:train_size]
val_data = dataset[train_size:train_size+val_size]
test_data = dataset[train_size+val_size:]

print(f"\nDataset généré:")
print(f"   Train: {len(train_data)} exemples")
print(f"   Val: {len(val_data)} exemples")
print(f"   Test: {len(test_data)} exemples")
print(f"   Total: {len(dataset)} exemples")

# Sauvegarder
data_dir = r"C:\Users\T9Y\llm-learning-journey\data"
os.makedirs(data_dir, exist_ok=True)

with open(os.path.join(data_dir, 'mom_incidents_train.json'), "w") as f:
    json.dump(train_data, f, indent=2)

with open(os.path.join(data_dir, 'mom_incidents_val.json'), "w") as f:
    json.dump(val_data, f, indent=2)

with open(os.path.join(data_dir, 'mom_incidents_test.json'), "w") as f:
    json.dump(test_data, f, indent=2)

print(f"\nFichiers sauvegardés")

# Statistiques
from collections import Counter
dist = Counter([item["label"] for item in train_data])
print("\nDistribution (Train):")
for label, count in sorted(dist.items()):
    print(f"   {label}: {count} exemples")

print("\nDataset prêt pour réentraînement")
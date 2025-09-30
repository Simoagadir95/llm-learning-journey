# 01_generate_training_data.py
"""
Génération d'un dataset d'entraînement pour classification d'incidents MOM
"""

import json
import random
import os

print("🏭 Génération du dataset d'incidents MOM...")

# Dataset manuel avec 100 exemples d'incidents manufacturing
categories = {
    "MACHINE_BREAKDOWN": [
        "CNC-A3 stopped unexpectedly with error E402. Hydraulic pressure low detected.",
        "Robotic arm #7 experienced servo motor failure. Production halted on Line 2.",
        "Conveyor belt system malfunction. Emergency stop activated at 14:32.",
        "Injection molding machine temperature sensor failure. Error code E501.",
        "Press machine hydraulic leak detected. Oil puddle under unit.",
        "Spindle bearing seized on CNC-B5. Unusual grinding noise reported.",
        "PLC communication error on Assembly Line 3. All stations offline.",
        "Pneumatic actuator stuck in extended position. Manual override required.",
        "Cooling system failure on Furnace #2. Temperature rising above limits.",
        "Vision system camera malfunction. Quality inspection offline.",
        "Robot controller error E303. Cannot complete pick-and-place cycle.",
        "Hydraulic pump cavitation detected on Press #4.",
        "Encoder failure on Motor Drive 12. Position feedback lost.",
        "Emergency stop circuit triggered unexpectedly. All lines down.",
        "Gearbox oil leak on Conveyor Drive Unit 5.",
        "Laser cutting head collision with material. Optical sensor damaged.",
        "Vacuum pump failure on Packaging Line. Low pressure alarm.",
        "Welding torch electrode wear exceeded threshold. Replacement needed.",
        "Lubrication system blocked on CNC-C2. Alarm triggered.",
        "Power supply failure on PLC cabinet. Backup battery depleted."
    ],
    
    "QUALITY_ISSUE": [
        "Quality inspection failed on Batch B2045. 15% of units outside tolerance.",
        "Dimensional variance detected. Parts measuring 0.3mm over specification.",
        "Surface finish defects found on 23 units. Rough texture on coating.",
        "Color mismatch detected in powder coating. Batch A302 rejected.",
        "Weld penetration insufficient on 8 assemblies. X-ray inspection failed.",
        "Thread depth non-conformance. Tap tool wear suspected.",
        "Burr height exceeds specification on machined edges.",
        "Material hardness test failure. Below HRC 45 requirement.",
        "Paint thickness uneven. Orange peel effect visible.",
        "Assembly torque values outside range. 12 fasteners under spec.",
        "Leak test failure on 5 housings. Pressure drop detected.",
        "Optical inspection detected scratches on polished surfaces.",
        "Contamination found in sealed components. Foreign particles present.",
        "Concentricity error exceeds 0.05mm tolerance.",
        "Material traceability issue. Heat lot number illegible.",
        "Package seal strength below minimum. 3 samples failed peel test.",
        "Label placement accuracy outside acceptable range.",
        "Functional test failure. 4 units show intermittent operation.",
        "Chemical composition analysis out of specification.",
        "Statistical process control alert. Cpk dropped below 1.33."
    ],
    
    "SAFETY_ISSUE": [
        "Operator reported safety guard malfunction on Press #3.",
        "Light curtain bypass detected. Safety system integrity compromised.",
        "Emergency stop button unresponsive. Immediate inspection required.",
        "Forklift near-miss incident in Warehouse Zone B. No injuries.",
        "Chemical spill in mixing area. 2 liters of solvent on floor.",
        "Lockout/Tagout procedure violation observed during maintenance.",
        "Electrical panel door found open during operation. Exposed contacts.",
        "Personal protective equipment non-compliance. 2 operators without safety glasses.",
        "Slip hazard reported near degreasing station. Oil residue on floor.",
        "Compressed air hose disconnected under pressure. Whipping hazard.",
        "Fire extinguisher inspection overdue by 3 months.",
        "Machine guarding missing on rotating shaft. Pinch point exposed.",
        "Ergonomic concern reported. Repetitive motion causing strain.",
        "Fall from height near-miss. Ladder placement unstable.",
        "Ventilation system inadequate. Fume concentration above limit.",
        "Hot surface warning label missing on oven door.",
        "Hearing protection non-compliance in high noise area.",
        "First aid kit supplies depleted. Bandages and burn gel needed.",
        "Aisle blocked with materials. Emergency exit obstructed.",
        "Pressure relief valve test overdue. Last inspection 18 months ago."
    ],
    
    "MAINTENANCE": [
        "Scheduled preventive maintenance completed on CNC-A3. All systems operational.",
        "Lubrication performed on Conveyor Drive Units 1-5. Next service due 2025-02-15.",
        "Filter replacement completed on Hydraulic Power Unit. Pressure normalized.",
        "Calibration of torque wrenches completed. Certificates updated.",
        "Bearing replacement on Motor Drive 7. Vibration levels normal.",
        "Belt tension adjustment on Conveyor System. Tracking improved.",
        "Sensor cleaning performed on Vision Inspection Station.",
        "Coolant system flushed and refilled. Concentration at 8%.",
        "Air filter replacement on Compressor #2. Pressure recovery good.",
        "Grease application on all linear guides per PM schedule.",
        "Battery replacement in UPS system. Backup power tested OK.",
        "Alignment check performed on robotic arm. Within specification.",
        "Infrared thermography scan completed. No hot spots detected.",
        "Vibration analysis on all rotating equipment. Baseline updated.",
        "Electrical connections tightened per thermal inspection.",
        "Hydraulic fluid analysis results normal. No contamination.",
        "Pneumatic line pressure test completed. No leaks found.",
        "Software update applied to PLC controllers. Version 4.2 installed.",
        "Emergency lighting test completed. All fixtures functional.",
        "Safety interlock function test passed. Response time 45ms."
    ],
    
    "SUPPLY_CHAIN": [
        "Material shortage detected. Steel stock below reorder point.",
        "Supplier delivery delay. Raw materials expected 3 days late.",
        "Quality issue with incoming components. Batch rejected by receiving inspection.",
        "Packaging materials out of stock. Production delayed 2 hours.",
        "Wrong part number delivered. PO-2045 items do not match specification.",
        "Freight damage reported on incoming shipment. 15 pallets affected.",
        "Supplier change notification received. New source for Part #A402.",
        "Inventory discrepancy found. Physical count 50 units short.",
        "Urgent material request for Batch B3021. Stock depleted earlier than forecast.",
        "Lead time increased by supplier. 6 weeks instead of 4 weeks.",
        "Raw material certificate of conformance missing from shipment.",
        "Alternative supplier approved due to capacity constraints.",
        "Expedited shipping required. Additional cost $850 approved.",
        "Material handling damage during internal transfer. 8 units scrapped.",
        "Warehouse space limitation. Cannot receive scheduled delivery.",
        "Price increase notification from supplier. 12% effective next month.",
        "Minimum order quantity changed. New MOQ is 500 units.",
        "Consignment inventory agreement signed. Stock level maintained.",
        "Just-in-time delivery sequence disruption. Line feed shortage.",
        "Import customs clearance delayed. Documents missing."
    ]
}

# Générer le dataset
dataset = []
for category, examples in categories.items():
    for text in examples:
        dataset.append({
            "text": text,
            "label": category
        })

# Mélanger aléatoirement
random.shuffle(dataset)

# Séparer train/validation/test (70/15/15)
n = len(dataset)
train_size = int(0.7 * n)
val_size = int(0.15 * n)

train_data = dataset[:train_size]
val_data = dataset[train_size:train_size+val_size]
test_data = dataset[train_size+val_size:]

print(f"✅ Dataset généré:")
print(f"   📊 Train: {len(train_data)} exemples")
print(f"   📊 Validation: {len(val_data)} exemples")
print(f"   📊 Test: {len(test_data)} exemples")
print(f"   📊 Total: {len(dataset)} exemples")

# Chemins absolus
data_dir = r"C:\Users\T9Y\llm-learning-journey\data"
os.makedirs(data_dir, exist_ok=True)

train_path = os.path.join(data_dir, 'mom_incidents_train.json')
val_path = os.path.join(data_dir, 'mom_incidents_val.json')
test_path = os.path.join(data_dir, 'mom_incidents_test.json')

print(f"\n💾 Sauvegarde des fichiers...")
print(f"   Dossier cible: {data_dir}")

# Sauvegarde avec gestion d'erreurs
try:
    with open(train_path, "w", encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Train: {train_path}")
    
    with open(val_path, "w", encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Val: {val_path}")
    
    with open(test_path, "w", encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Test: {test_path}")
    
    # Vérification immédiate
    print(f"\n🔍 Vérification:")
    if os.path.exists(train_path):
        size = os.path.getsize(train_path)
        print(f"   ✅ Train créé ({size} bytes)")
    else:
        print(f"   ❌ Train non créé!")
        
    if os.path.exists(val_path):
        size = os.path.getsize(val_path)
        print(f"   ✅ Val créé ({size} bytes)")
    else:
        print(f"   ❌ Val non créé!")
        
    if os.path.exists(test_path):
        size = os.path.getsize(test_path)
        print(f"   ✅ Test créé ({size} bytes)")
    else:
        print(f"   ❌ Test non créé!")

except PermissionError as e:
    print(f"❌ ERREUR: Pas de permissions d'écriture!")
    print(f"   {e}")
except Exception as e:
    print(f"❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()

# Statistiques par catégorie
from collections import Counter
train_dist = Counter([item["label"] for item in train_data])
print("\n📊 Distribution des catégories (Train):")
for label, count in sorted(train_dist.items()):
    print(f"   {label}: {count} exemples")

print("\n🎉 Script terminé!")
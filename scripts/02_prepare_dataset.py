# 02_prepare_dataset.py
"""
Conversion du dataset JSON en format Hugging Face Datasets
"""

from datasets import Dataset, DatasetDict
import json
import os

print("🔄 Préparation du dataset pour Hugging Face...")

# Chemins absolus
data_dir = r"C:\Users\T9Y\llm-learning-journey\data"

# Charger les données
print("\n📂 Chargement des fichiers JSON...")
with open(os.path.join(data_dir, "mom_incidents_train.json"), "r") as f:
    train_data = json.load(f)

with open(os.path.join(data_dir, "mom_incidents_val.json"), "r") as f:
    val_data = json.load(f)

with open(os.path.join(data_dir, "mom_incidents_test.json"), "r") as f:
    test_data = json.load(f)

print(f"✅ Fichiers chargés:")
print(f"   Train: {len(train_data)} exemples")
print(f"   Val: {len(val_data)} exemples")
print(f"   Test: {len(test_data)} exemples")

# Créer les datasets Hugging Face
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)

# Créer un DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

print("\n✅ Datasets Hugging Face créés:")
print(dataset_dict)

# Mapper les labels vers des IDs
label2id = {
    "MACHINE_BREAKDOWN": 0,
    "QUALITY_ISSUE": 1,
    "SAFETY_ISSUE": 2,
    "MAINTENANCE": 3,
    "SUPPLY_CHAIN": 4
}

id2label = {v: k for k, v in label2id.items()}

print("\n🏷️  Label Mapping:")
for label, id in label2id.items():
    print(f"   {label} → {id}")

def preprocess_function(examples):
    """Convertir les labels en IDs"""
    examples["labels"] = [label2id[label] for label in examples["label"]]
    return examples

# Appliquer la transformation
dataset_dict = dataset_dict.map(preprocess_function, batched=True)

# Sauvegarder
dataset_path = os.path.join(data_dir, "mom_incidents_dataset")
dataset_dict.save_to_disk(dataset_path)

print(f"\n✅ Dataset sauvegardé dans: {dataset_path}")
print(f"\n📊 Exemple du dataset:")
print(dataset_dict["train"][0])

# Sauvegarder les mappings
mappings_path = os.path.join(data_dir, "label_mappings.json")
with open(mappings_path, "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

print(f"\n✅ Mappings sauvegardés dans: {mappings_path}")
print("\n🎯 Dataset prêt pour le fine-tuning !")
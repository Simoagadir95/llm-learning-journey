# 03_finetune_model.py
"""
Fine-tuning de DistilBERT pour classification d'incidents MOM
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
import os

print("🏭 MOM - Fine-tuning du classificateur d'incidents")
print("=" * 70)

# Chemins
data_dir = r"C:\Users\T9Y\llm-learning-journey\data"
models_dir = r"C:\Users\T9Y\llm-learning-journey\models"
os.makedirs(models_dir, exist_ok=True)

# 1. Charger le dataset
print("\n📂 Chargement du dataset...")
dataset = load_from_disk(os.path.join(data_dir, "mom_incidents_dataset"))

# Charger les mappings
with open(os.path.join(data_dir, "label_mappings.json"), "r") as f:
    mappings = json.load(f)
    label2id = mappings["label2id"]
    id2label = {int(k): v for k, v in mappings["id2label"].items()}

print(f"✅ Dataset chargé:")
print(f"   Train: {len(dataset['train'])} exemples")
print(f"   Val: {len(dataset['validation'])} exemples")
print(f"   Test: {len(dataset['test'])} exemples")

# 2. Charger tokenizer et modèle
print("\n🤖 Chargement du modèle DistilBERT...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,
    id2label=id2label,
    label2id=label2id
)

print(f"✅ Modèle chargé: {model.num_parameters():,} paramètres")

# 3. Tokenisation
print("\n🔤 Tokenisation des données...")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# AJOUTEZ CES LIGNES ICI :
# Supprimer les colonnes inutiles (garder seulement celles nécessaires pour l'entraînement)
tokenized_dataset = tokenized_dataset.remove_columns(["text", "label"])

print("✅ Tokenisation terminée")

# 4. Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5. Métriques d'évaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# 6. Configuration de l'entraînement
print("\n⚙️  Configuration de l'entraînement...")

training_args = TrainingArguments(
    output_dir=os.path.join(models_dir, "mom_incident_classifier"),
    eval_strategy="epoch",  # Changé: evaluation_strategy → eval_strategy
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir=os.path.join(models_dir, "logs"),
    logging_steps=5,
    save_total_limit=2,
    push_to_hub=False
)

print(f"✅ Configuration:")
print(f"   📊 Batch size: {training_args.per_device_train_batch_size}")
print(f"   📈 Learning rate: {training_args.learning_rate}")
print(f"   🔄 Epochs: {training_args.num_train_epochs}")
print(f"   💾 Output: {training_args.output_dir}")

# 7. Création du Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 8. Entraînement
print("\n🚀 DÉBUT DE L'ENTRAÎNEMENT...")
print("=" * 70)
print("⏰ Cela peut prendre 10-15 minutes...")
print()

train_result = trainer.train()

print("\n✅ ENTRAÎNEMENT TERMINÉ!")
print("=" * 70)

# 9. Évaluation finale
print("\n📊 ÉVALUATION SUR TEST SET...")

test_results = trainer.evaluate(tokenized_dataset["test"])

print("\n🎯 RÉSULTATS FINAUX:")
print(f"   Accuracy: {test_results['eval_accuracy']:.4f} ({test_results['eval_accuracy']*100:.2f}%)")
print(f"   F1 Score: {test_results['eval_f1']:.4f}")
print(f"   Precision: {test_results['eval_precision']:.4f}")
print(f"   Recall: {test_results['eval_recall']:.4f}")

# 10. Sauvegarde du modèle final
print("\n💾 Sauvegarde du modèle final...")

final_model_path = os.path.join(models_dir, "mom_incident_classifier_final")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"✅ Modèle sauvegardé dans: {final_model_path}")

# 11. Matrice de confusion
print("\n📈 Génération de la matrice de confusion...")

predictions = trainer.predict(tokenized_dataset["test"])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

cm = confusion_matrix(y_true, y_pred)

print("\n🔢 MATRICE DE CONFUSION:")
print("Colonnes: Prédictions | Lignes: Réalité")
print("\n      ", end="")
for i in range(5):
    print(f"{id2label[i][:10]:12s}", end="")
print()
for i, row in enumerate(cm):
    print(f"{id2label[i][:10]:10s}", row)

print("\n🎉 FINE-TUNING TERMINÉ AVEC SUCCÈS!")
print("=" * 70)
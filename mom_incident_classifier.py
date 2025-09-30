# mom_incident_classifier.py
"""
Classification automatique des incidents de production
Cas d'usage: Triage automatique des tickets dans un système MOM
"""

from transformers import pipeline
import torch

print("🏭 MOM - Classificateur d'incidents de production\n")

# Modèle de classification (sentiment = exemple simple, on fera mieux plus tard)
classifier = pipeline("text-classification")

# Exemples de rapports d'incidents réels dans une usine
incidents = [
    {
        "id": "INC-001",
        "text": "Machine #3 stopped unexpectedly. Error code E402. Production halted.",
        "expected": "Machine Breakdown"
    },
    {
        "id": "INC-002", 
        "text": "Quality check failed. 15% of units outside tolerance. Batch #A2047.",
        "expected": "Quality Issue"
    },
    {
        "id": "INC-003",
        "text": "Operator reported safety concern. Emergency stop button malfunction.",
        "expected": "Safety Issue"
    },
    {
        "id": "INC-004",
        "text": "Scheduled maintenance completed. All systems operational.",
        "expected": "Maintenance Report"
    },
    {
        "id": "INC-005",
        "text": "Material shortage detected. Production delayed by 2 hours.",
        "expected": "Supply Chain"
    }
]

# Classification
print("📊 CLASSIFICATION DES INCIDENTS:\n")
for incident in incidents:
    # Note: On utilise un modèle générique pour l'instant
    # Plus tard, on fine-tunera un modèle spécialisé pour le manufacturing
    result = classifier(incident["text"])[0]
    
    print(f"🎫 {incident['id']}")
    print(f"   📝 Texte: {incident['text']}")
    print(f"   🏷️  Attendu: {incident['expected']}")
    print(f"   🤖 Prédiction modèle: {result['label']} (confiance: {result['score']:.2%})")
    print()

print("💡 Note: Ce modèle n'est pas entraîné pour le manufacturing.")
print("   Dans Pratique-002, nous allons fine-tuner un modèle spécialisé!")
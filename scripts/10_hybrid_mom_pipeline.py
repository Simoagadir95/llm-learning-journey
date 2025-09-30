# 10_hybrid_mom_pipeline.py
"""
Pipeline MOM hybride : Classification + RAG
"""

import chromadb
from transformers import pipeline
import os

print("Initialisation du système MOM hybride...")
print("=" * 70)

# 1. Classificateur (même cassé, on l'utilise pour la démo d'architecture)
models_dir = r"C:\Users\T9Y\llm-learning-journey\models"
classifier_path = os.path.join(models_dir, "mom_incident_classifier_final")

try:
    classifier = pipeline("text-classification", model=classifier_path)
    print("Classificateur chargé")
except:
    print("Classificateur non disponible, utilisation du fallback")
    classifier = None

# 2. Base vectorielle RAG
vectorstore_path = r"C:\Users\T9Y\llm-learning-journey\vectorstore"
client = chromadb.PersistentClient(path=vectorstore_path)
collection = client.get_collection(name="mom_knowledge_base")
print(f"Base vectorielle chargée: {collection.count()} documents")

# 3. Générateur
generator = pipeline("text-generation", model="gpt2")
print("Générateur chargé")

print("\nSystème prêt")
print("=" * 70)

def process_incident(incident_text):
    """
    Pipeline complet:
    1. Classification de l'incident
    2. Recherche de procédures pertinentes (RAG)
    3. Génération de recommandations
    """
    
    print(f"\nIncident: {incident_text}")
    print("-" * 70)
    
    # Étape 1: Classification
    if classifier:
        category_result = classifier(incident_text)[0]
        category = category_result['label']
        confidence = category_result['score']
        print(f"Catégorie: {category} ({confidence:.1%})")
    else:
        category = "UNKNOWN"
        confidence = 0.0
        print("Catégorie: Non classifié")
    
    # Étape 2: Recherche RAG
    search_query = f"{category}: {incident_text}"
    results = collection.query(
        query_texts=[search_query],
        n_results=2
    )
    
    print(f"Documents trouvés: {len(results['documents'][0])}")
    for i, metadata in enumerate(results['metadatas'][0], 1):
        print(f"  {i}. {metadata['source']} ({metadata['category']})")
    
    # Étape 3: Génération recommandations
    context = "\n".join(results['documents'][0])
    
    prompt = f"""Incident type: {category}

Relevant procedures:
{context[:500]}

Incident: {incident_text}

Recommended actions:"""
    
    response = generator(
        prompt,
        max_length=150,
        temperature=0.7,
        do_sample=True,
        pad_token_id=50256
    )[0]['generated_text']
    
    recommendation = response.split("Recommended actions:")[-1].strip()
    
    print(f"\nRecommandations:")
    print(f"  {recommendation[:200]}")
    
    print(f"\nSources: {', '.join([m['source'] for m in results['metadatas'][0]])}")
    
    return {
        "category": category,
        "confidence": confidence,
        "recommendation": recommendation,
        "sources": [m['source'] for m in results['metadatas'][0]]
    }

# Tests
incidents = [
    "CNC-A3 stopped with error E402, hydraulic pressure low",
    "Quality inspection failed, 15 units with surface defects",
    "Operator slipped near coolant station, no injury but safety concern",
    "Preventive maintenance completed on all conveyors",
    "Steel raw material delayed, supplier cited transport issues"
]

print("\nTEST DU SYSTÈME HYBRIDE")
print("=" * 70)

for incident in incidents:
    result = process_incident(incident)
    print("=" * 70)

print("\nSystème hybride fonctionnel!")
print("Classification → RAG → Recommandations avec sources")
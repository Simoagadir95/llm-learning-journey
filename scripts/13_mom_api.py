# 13_mom_api.py
"""
API REST pour le système MOM
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import chromadb
from transformers import pipeline
import os
import uvicorn

print("Initialisation de l'API MOM...")

app = FastAPI(
    title="MOM LLM API",
    description="API pour classification d'incidents et assistance opérateur",
    version="1.0.0"
)

# Charger les composants
models_dir = r"C:\Users\T9Y\llm-learning-journey\models"
vectorstore_path = r"C:\Users\T9Y\llm-learning-journey\vectorstore"

# Classificateur
try:
    classifier_path = os.path.join(models_dir, "mom_incident_classifier_final")
    classifier = pipeline("text-classification", model=classifier_path)
    print("Classificateur chargé")
except:
    classifier = None
    print("Classificateur non disponible")

# RAG
client = chromadb.PersistentClient(path=vectorstore_path)
collection = client.get_collection(name="mom_knowledge_base")
generator = pipeline("text-generation", model="gpt2")

print("Système chargé\n")

# Modèles de données
class Incident(BaseModel):
    text: str
    machine_id: Optional[str] = None
    timestamp: Optional[str] = None

class Question(BaseModel):
    question: str
    top_k: Optional[int] = 2

class ClassificationResponse(BaseModel):
    category: str
    confidence: float
    text: str

class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]

class IncidentResponse(BaseModel):
    category: str
    confidence: float
    recommendation: str
    sources: list[str]

# Endpoints
@app.get("/")
def read_root():
    return {
        "message": "MOM LLM API",
        "status": "operational",
        "endpoints": [
            "/classify",
            "/ask",
            "/process-incident"
        ]
    }

@app.post("/classify", response_model=ClassificationResponse)
def classify_incident(incident: Incident):
    """Classifier un incident"""
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not available")
    
    result = classifier(incident.text)[0]
    
    return ClassificationResponse(
        category=result['label'],
        confidence=result['score'],
        text=incident.text
    )

@app.post("/ask", response_model=RAGResponse)
def ask_question(question: Question):
    """Répondre à une question avec RAG"""
    
    # Recherche
    results = collection.query(
        query_texts=[question.question],
        n_results=question.top_k
    )
    
    # Génération
    context = "\n\n".join(results['documents'][0])
    prompt = f"""Manufacturing procedures:

{context}

Question: {question.question}

Answer:"""
    
    response = generator(
        prompt,
        max_length=150,
        temperature=0.7,
        do_sample=True,
        pad_token_id=50256
    )[0]['generated_text']
    
    answer = response.split("Answer:")[-1].strip() if "Answer:" in response else response
    sources = [m['source'] for m in results['metadatas'][0]]
    
    return RAGResponse(
        question=question.question,
        answer=answer,
        sources=sources
    )

@app.post("/process-incident", response_model=IncidentResponse)
def process_incident(incident: Incident):
    """Pipeline complet: classification + RAG + recommandations"""
    
    # Classification
    if classifier:
        cat_result = classifier(incident.text)[0]
        category = cat_result['label']
        confidence = cat_result['score']
    else:
        category = "UNKNOWN"
        confidence = 0.0
    
    # RAG
    search_query = f"{category}: {incident.text}"
    results = collection.query(
        query_texts=[search_query],
        n_results=2
    )
    
    # Génération
    context = "\n".join(results['documents'][0])
    prompt = f"""Incident type: {category}

Procedures:
{context[:500]}

Incident: {incident.text}

Recommended actions:"""
    
    response = generator(
        prompt,
        max_length=150,
        temperature=0.7,
        do_sample=True,
        pad_token_id=50256
    )[0]['generated_text']
    
    recommendation = response.split("Recommended actions:")[-1].strip()
    sources = [m['source'] for m in results['metadatas'][0]]
    
    return IncidentResponse(
        category=category,
        confidence=confidence,
        recommendation=recommendation,
        sources=sources
    )

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "classifier": "available" if classifier else "unavailable",
        "rag_documents": collection.count()
    }

if __name__ == "__main__":
    print("Démarrage de l'API sur http://localhost:8001")
    print("Documentation: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Changé de 8000 à 8001
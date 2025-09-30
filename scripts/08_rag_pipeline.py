# 08_rag_pipeline.py
"""
Pipeline RAG simple pour MOM
"""

import chromadb
from transformers import pipeline
import os

print("Initialisation du pipeline RAG...")

# Charger la base vectorielle
vectorstore_path = r"C:\Users\T9Y\llm-learning-journey\vectorstore"
client = chromadb.PersistentClient(path=vectorstore_path)
collection = client.get_collection(name="mom_knowledge_base")

print(f"Base chargée: {collection.count()} documents")

# Modèle de génération (GPT-2 pour démo)
generator = pipeline("text-generation", model="gpt2")

def rag_query(question, top_k=2):
    """
    Pipeline RAG complet:
    1. Recherche documents pertinents
    2. Construction du contexte
    3. Génération de la réponse
    """
    
    print(f"\nQuestion: {question}")
    print("-" * 60)
    
    # 1. Recherche
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    
    # 2. Contexte
    print("Documents trouvés:")
    contexts = []
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n{i+1}. Source: {metadata['source']}")
        print(f"   Catégorie: {metadata['category']}")
        print(f"   Extrait: {doc[:150]}...")
        contexts.append(doc)
    
    context = "\n\n".join(contexts)
    
    # 3. Génération
    prompt = f"""Based on the following manufacturing procedures:

{context}

Question: {question}

Answer:"""
    
    print("\n" + "=" * 60)
    print("Génération de la réponse...")
    
    response = generator(
        prompt,
        max_length=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=50256
    )[0]['generated_text']
    
    # Extraire juste la réponse (après "Answer:")
    answer = response.split("Answer:")[-1].strip()
    
    print(f"\nRéponse générée:")
    print(answer)
    print("=" * 60)
    
    return {
        "question": question,
        "answer": answer,
        "sources": [m['source'] for m in results['metadatas'][0]]
    }

# Tests
questions = [
    "How to start CNC-A3 machine?",
    "What to do for error E402?",
    "What is the procedure when quality check fails?"
]

print("\n" + "=" * 60)
print("TEST DU PIPELINE RAG")
print("=" * 60)

for question in questions:
    result = rag_query(question)
    print(f"\nSources utilisées: {', '.join(result['sources'])}")
    print("\n" + "=" * 60)

print("\nPipeline RAG fonctionnel!")
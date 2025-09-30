# 09_rag_operator_assistant.py
"""
Assistant opérateur MOM avec RAG
Comparaison avec génération pure (Script 4)
"""

import chromadb
from transformers import pipeline

print("🏭 MOM - Assistant Opérateur avec RAG")
print("=" * 70)

# Charger la base vectorielle
vectorstore_path = r"C:\Users\T9Y\llm-learning-journey\vectorstore"
client = chromadb.PersistentClient(path=vectorstore_path)
collection = client.get_collection(name="mom_knowledge_base")

print(f"Base chargée: {collection.count()} documents\n")

# Générateur
generator = pipeline("text-generation", model="gpt2")

def answer_with_rag(question):
    """Répondre avec RAG"""
    # Recherche
    results = collection.query(
        query_texts=[question],
        n_results=2
    )
    
    # Contexte
    context = "\n\n".join(results['documents'][0])
    sources = [m['source'] for m in results['metadatas'][0]]
    
    # Prompt enrichi
    prompt = f"""Manufacturing procedure:

{context}

Operator question: {question}
Answer:"""
    
    response = generator(
        prompt,
        max_length=150,
        temperature=0.7,
        do_sample=True,
        pad_token_id=50256
    )[0]['generated_text']
    
    answer = response.split("Answer:")[-1].strip() if "Answer:" in response else response.split(question)[-1].strip()
    
    return answer, sources

def answer_without_rag(question):
    """Répondre sans RAG (comme Script 4)"""
    prompt = f"Manufacturing FAQ: {question}\nAnswer:"
    
    response = generator(
        prompt,
        max_length=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=50256
    )[0]['generated_text']
    
    answer = response.split("Answer:")[-1].strip()
    return answer

# Questions test
questions = [
    "How to start CNC-A3 machine?",
    "What should I do for error E402?",
    "What is the procedure when quality check fails?"
]

print("COMPARAISON : Sans RAG vs Avec RAG")
print("=" * 70)

for i, question in enumerate(questions, 1):
    print(f"\n{i}. QUESTION: {question}")
    print("-" * 70)
    
    # Sans RAG
    print("❌ SANS RAG (génération pure):")
    answer_no_rag = answer_without_rag(question)
    print(f"   {answer_no_rag[:200]}")
    
    # Avec RAG
    print("\n✅ AVEC RAG (recherche + génération):")
    answer_rag, sources = answer_with_rag(question)
    print(f"   {answer_rag[:200]}")
    print(f"   📚 Sources: {', '.join(sources)}")
    print()

print("=" * 70)
print("\nRésultat: Le RAG ancre les réponses dans des documents réels!")
print("Les réponses sont vérifiables et traçables.")
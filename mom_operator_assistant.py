# mom_operator_assistant.py
"""
Assistant virtuel pour opérateurs de production
Cas d'usage: Support technique en temps réel
"""

from transformers import pipeline

print("🏭 MOM - Assistant Virtuel Opérateur\n")

# Générateur de texte (conversationnel)
generator = pipeline("text-generation", model="gpt2")

# Base de connaissances (FAQ typiques en manufacturing)
knowledge_base = """
Common Manufacturing Procedures:

Machine Startup:
1. Check safety guards are in place
2. Verify material supply is adequate
3. Run diagnostic cycle
4. Confirm all sensors show green status
5. Begin production cycle

Error Code E402 - Hydraulic Pressure Low:
- Check hydraulic fluid level
- Inspect pump operation
- Verify pressure sensor readings
- If pressure < 3000 PSI, call maintenance
- Do not restart until issue resolved

Quality Check Failure:
- Stop production immediately
- Tag affected batch
- Notify quality supervisor
- Document in quality log
- Wait for inspection before resuming
"""

# Questions typiques d'opérateurs
operator_questions = [
    "How to start Machine CNC-A3?",
    "What to do for error E402?",
    "Quality check failed, what are the steps?"
]

print("📚 BASE DE CONNAISSANCES CHARGÉE")
print(f"   {len(knowledge_base.split())} mots de procédures\n")

print("❓ SIMULATION DE QUESTIONS OPÉRATEURS:\n")

for i, question in enumerate(operator_questions, 1):
    print(f"{i}. OPÉRATEUR: {question}")
    
    # Contexte + Question
    prompt = f"Manufacturing FAQ: {question}\nAnswer:"
    
    response = generator(
        prompt,
        max_length=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=50256
    )[0]['generated_text']
    
    # Extraire juste la réponse (après "Answer:")
    answer = response.split("Answer:")[-1].strip()
    
    print(f"   🤖 ASSISTANT: {answer}")
    print()

print("💡 Note: Ce modèle n'est pas spécialisé en manufacturing.")
print("   Un modèle fine-tuné sur vos procédures serait bien plus précis!")
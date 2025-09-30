# mom_log_parser.py
"""
Extraction d'informations structurées des logs machines
Cas d'usage: Parsing intelligent de logs non structurés
"""

from transformers import pipeline

print("🏭 MOM - Analyseur de logs machines\n")

# Pipeline de Question-Réponse
qa_pipeline = pipeline("question-answering")

# Exemple de log machine (format réel d'une ligne de production)
machine_log = """
[2025-01-15 14:32:45] Machine ID: CNC-A3 | Status: ERROR
Error Code: E402 - Hydraulic Pressure Low
Production Line: Assembly Line 2
Downtime Duration: 45 minutes
Root Cause: Hydraulic pump failure detected at sensor S12
Corrective Action: Pump replaced, system restarted at 15:17:45
Parts Affected: Batch B2045 (127 units) - Quality inspection required
Operator: John Smith (ID: OP-1847)
Shift: Day Shift - Team Alpha
"""

# Questions qu'un système MOM devrait extraire automatiquement
questions = [
    "What is the error code?",
    "How long was the downtime?",
    "What was the root cause?",
    "Which batch was affected?",
    "Who was the operator?",
    "What corrective action was taken?"
]

print("📋 LOG MACHINE ANALYSÉ:")
print(machine_log)
print("\n🔍 EXTRACTION AUTOMATIQUE D'INFORMATIONS:\n")

for question in questions:
    result = qa_pipeline(question=question, context=machine_log)
    print(f"❓ {question}")
    print(f"✅ {result['answer']} (confiance: {result['score']:.2%})")
    print()

print("💡 Application MOM: Ces données peuvent être automatiquement")
print("   insérées dans une base de données pour analyse et reporting.")
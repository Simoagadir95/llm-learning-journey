# 06_install_rag_dependencies.py
"""
Installation des dépendances pour RAG
"""
import subprocess
import sys

print("Installation des packages RAG...")

packages = [
    "chromadb",
    "sentence-transformers",
    "langchain",
    "langchain-community"
]

for package in packages:
    print(f"\nInstallation de {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("\nTous les packages sont installés!")

# Vérification
print("\nVérification des imports...")
try:
    import chromadb
    print("✅ chromadb")
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers")
    import langchain
    print("✅ langchain")
    print("\nTout est prêt pour le RAG!")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
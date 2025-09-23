import torch
import transformers
print("✅ VS Code avec environnement llm-learning")
print(f"✅ PyTorch: {torch.__version__}")  
print(f"✅ Transformers: {transformers.__version__}")

# Test d'un modèle simple
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("VS Code est maintenant configuré!")
print(f"✅ Test modèle: {result}")

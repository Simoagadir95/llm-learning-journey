from transformers import pipeline
import os

models_dir = r"C:\Users\T9Y\llm-learning-journey\models"
model_path = os.path.join(models_dir, "mom_incident_classifier_final")

print("Test du modèle fine-tuné...\n")

classifier = pipeline("text-classification", model=model_path)

# Test sur différents types
tests = [
    "CNC machine stopped with hydraulic error E402",
    "Quality inspection failed, 20% defects detected",
    "Safety guard malfunction reported",
    "Preventive maintenance completed successfully",
    "Material shortage, supplier delayed delivery"
]

for text in tests:
    result = classifier(text)[0]
    print(f"Text: {text}")
    print(f"→ {result['label']} ({result['score']:.2%})\n")
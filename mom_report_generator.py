# mom_report_generator.py
"""
Génération automatique de rapports de production
Cas d'usage: Synthèse quotidienne pour management
"""

from transformers import pipeline

print("🏭 MOM - Générateur de rapports de production\n")

# Pipeline de résumé
summarizer = pipeline("summarization")

# Données brutes de production (format verbeux)
production_data = """
Production Report - Assembly Line 2 - January 15, 2025

Shift A (06:00-14:00):
- Total units produced: 847 units
- Quality pass rate: 98.2%
- Machine CNC-A3 experienced hydraulic failure at 14:32, downtime 45 minutes
- 17 units failed quality inspection due to dimensional tolerance issues
- Average cycle time: 4.2 minutes per unit
- Material consumption: 1,240 kg steel, 340 kg aluminum
- 3 operators present, all safety protocols followed

Shift B (14:00-22:00):
- Total units produced: 892 units
- Quality pass rate: 99.1%
- No major incidents reported
- Minor calibration adjustment on Machine CNC-B1 at 18:15
- Average cycle time: 3.9 minutes per unit
- Material consumption: 1,305 kg steel, 356 kg aluminum
- 3 operators present, one near-miss safety incident logged and resolved

Overall Performance:
Production exceeded daily target of 1,600 units by 8.7%. Quality rates 
remained within acceptable range. The hydraulic failure on CNC-A3 requires 
preventive maintenance review. Material consumption is tracking 3% below 
forecast. Recommend continuing current production parameters for Shift A 
and B. Safety training refresher scheduled for next week following near-miss.
"""

print("📄 DONNÉES BRUTES DE PRODUCTION:")
print(f"   {len(production_data.split())} mots, {len(production_data)} caractères\n")

# Génération du résumé
summary = summarizer(
    production_data, 
    max_length=130, 
    min_length=50,
    do_sample=False
)[0]

print("📝 RÉSUMÉ EXÉCUTIF GÉNÉRÉ PAR IA:\n")
print(summary['summary_text'])

print("\n💡 Application MOM: Ce résumé peut être envoyé automatiquement")
print("   aux managers chaque jour par email ou dashboard.")
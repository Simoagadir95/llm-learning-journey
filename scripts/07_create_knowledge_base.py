# 07_create_knowledge_base.py
"""
Création d'une base de connaissances MOM avec ChromaDB
"""

import chromadb
from chromadb.config import Settings
import os

print("Création de la base de connaissances MOM...")

# Chemin vers la base vectorielle
vectorstore_path = r"C:\Users\T9Y\llm-learning-journey\vectorstore"
os.makedirs(vectorstore_path, exist_ok=True)

# Initialiser ChromaDB
client = chromadb.PersistentClient(path=vectorstore_path)

# Créer ou récupérer la collection
collection = client.get_or_create_collection(
    name="mom_knowledge_base",
    metadata={"description": "Manufacturing operations management procedures and manuals"}
)

# Documents MOM (manuels, procédures, SOPs)
documents = [
    # Procédures de démarrage machines
    {
        "text": """Machine CNC-A3 Startup Procedure:
1. Verify all safety guards are properly installed and secured
2. Check hydraulic fluid level in reservoir (must be above MIN mark)
3. Inspect coolant system - concentration should be 8-10%
4. Power on main electrical panel
5. Wait for hydraulic system to pressurize (3000 PSI minimum)
6. Run diagnostic cycle (Press green button, wait for 3 beeps)
7. Load program and verify tool offsets
8. Perform dry run before production
9. Document startup in logbook with operator signature""",
        "metadata": {"source": "CNC-A3 Manual", "category": "startup", "machine": "CNC-A3"}
    },
    
    # Codes erreur
    {
        "text": """Error Code E402 - Hydraulic Pressure Low:
Symptoms: Machine stops, red alarm light, pressure gauge below 2500 PSI
Causes:
- Low hydraulic fluid level
- Pump failure or cavitation
- Leaking hydraulic lines
- Clogged filter
Troubleshooting Steps:
1. Check fluid level in hydraulic reservoir
2. Inspect all visible hydraulic lines for leaks
3. Check hydraulic filter - replace if clogged
4. Listen for unusual pump noise (cavitation)
5. Verify pressure sensor readings at gauge G12
6. If pressure below 3000 PSI after checks, call maintenance
Safety: Do NOT restart machine until issue resolved. Tag out per LOTO procedure.""",
        "metadata": {"source": "Error Code Manual", "category": "troubleshooting", "error_code": "E402"}
    },
    
    {
        "text": """Error Code E501 - Temperature Sensor Failure:
Symptoms: Temperature reading shows -999 or fluctuates wildly
Causes: Sensor disconnected, damaged wiring, failed sensor
Resolution: Replace temperature sensor, recalibrate system
Downtime: Approximately 30 minutes""",
        "metadata": {"source": "Error Code Manual", "category": "troubleshooting", "error_code": "E501"}
    },
    
    # Procédures qualité
    {
        "text": """Quality Check Failure Procedure:
When quality inspection fails:
1. STOP production immediately - press emergency stop
2. Tag the affected batch with red QC HOLD tag
3. Quarantine all units from the batch
4. Document in Quality Log: batch number, defect type, quantity affected
5. Notify Quality Supervisor immediately
6. Take 3 sample units for detailed inspection
7. Do NOT resume production until root cause identified
8. Quality supervisor must approve restart
Common defects: dimensional variance, surface defects, material issues""",
        "metadata": {"source": "Quality SOP-003", "category": "quality"}
    },
    
    # Sécurité
    {
        "text": """Safety Incident Response Procedure:
For any safety incident or near-miss:
1. Ensure immediate safety of all personnel
2. Provide first aid if needed
3. Secure the area - prevent others from entering
4. Do NOT move any equipment or materials
5. Notify Safety Supervisor immediately (Ext. 4567)
6. Document incident in Safety Log within 1 hour
7. Take photos if safe to do so
8. Preserve evidence until investigation complete
9. All witnesses must provide written statements
Emergency contacts: Safety: 4567, Medical: 4911, Security: 4000""",
        "metadata": {"source": "Safety SOP-001", "category": "safety"}
    },
    
    # Maintenance
    {
        "text": """Preventive Maintenance Schedule CNC-A3:
Daily: Visual inspection, check fluid levels, clean work area
Weekly: Lubricate linear guides, check belt tension, inspect tooling
Monthly: Replace hydraulic filter, calibrate sensors, vibration analysis
Quarterly: Full system diagnostic, bearing inspection, electrical connections
Annual: Major overhaul, replace wear parts, recertification
Maintenance must be documented in CMMS system with completion signature.""",
        "metadata": {"source": "Maintenance Schedule", "category": "maintenance", "machine": "CNC-A3"}
    },
    
    # Problèmes supply chain
    {
        "text": """Material Shortage Protocol:
When material stock falls below reorder point:
1. Check inventory system for accurate count
2. Verify reorder point settings are correct
3. Contact purchasing department immediately
4. Identify alternative materials if available
5. Calculate production impact (units, days)
6. Notify production planning
7. Update production schedule
8. Document shortage in Supply Chain Log
Lead times: Standard materials 2-3 weeks, Special materials 6-8 weeks""",
        "metadata": {"source": "Supply Chain SOP-012", "category": "supply_chain"}
    },
    
    # Procédures générales
    {
        "text": """Shift Handover Procedure:
At end of shift:
1. Complete production log with quantities produced
2. Document any issues or abnormalities
3. Note machine status and any pending maintenance
4. Brief incoming shift supervisor on current status
5. Transfer any open work orders
6. Ensure all safety incidents are logged
7. Clean work area and organize tools
8. Sign off in shift log""",
        "metadata": {"source": "Operations Manual", "category": "procedures"}
    },
    
    {
        "text": """Lockout/Tagout (LOTO) Procedure:
Before any maintenance or repair:
1. Notify affected personnel of shutdown
2. Shut down machine using normal stop procedure
3. Isolate all energy sources (electrical, hydraulic, pneumatic)
4. Lock out energy isolation devices
5. Attach personal LOTO tag with name and date
6. Attempt to start machine to verify isolation
7. Release stored energy (bleed hydraulics, discharge capacitors)
8. Verify zero energy state before beginning work
Only the person who applied LOTO may remove it.""",
        "metadata": {"source": "Safety Manual", "category": "safety"}
    },
]

print(f"Ajout de {len(documents)} documents à la base...")

# Préparer les données pour ChromaDB
ids = [f"doc_{i}" for i in range(len(documents))]
texts = [doc["text"] for doc in documents]
metadatas = [doc["metadata"] for doc in documents]

# Ajouter à la collection
collection.add(
    documents=texts,
    ids=ids,
    metadatas=metadatas
)

print(f"✅ {len(documents)} documents ajoutés à la base vectorielle")

# Statistiques
categories = {}
for doc in documents:
    cat = doc["metadata"]["category"]
    categories[cat] = categories.get(cat, 0) + 1

print("\nDistribution par catégorie:")
for cat, count in categories.items():
    print(f"   {cat}: {count} documents")

print(f"\nBase de connaissances sauvegardée dans: {vectorstore_path}")
print("Prêt pour le RAG!")
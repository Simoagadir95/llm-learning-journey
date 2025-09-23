import wandb

# Initialiser un projet de test
wandb.init(
    project="llm-learning-setup", 
    name="test-configuration",
    tags=["setup", "test"]
)

# Logger quelques métriques de test
wandb.log({
    "test_metric": 42,
    "pytorch_version": "2.8.0",
    "transformers_version": "4.56.2",
    "setup_complete": 1.0
})

print("✅ Test W&B réussi!")
print("🔗 Consultez vos résultats sur: https://wandb.ai/simoagadir/llm-learning-setup")

wandb.finish()
#run.py
#currently using for system testing.
from TMN_DataGen import DatasetGenerator
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load('configs/default_config.yaml')

# Prepare data
sentence_pairs = [
    ("The cat chases the mouse.", "The mouse is being chased by the cat."),
    ("The dog barks.", "The cat meows.")
]
labels = ["entails", "neutral"]

# Generate dataset
generator = DatasetGenerator(config)
generator.generate_dataset(
    sentence_pairs=sentence_pairs,
    labels=labels,
    output_path='data/processed/entailment_dataset.pkl'
)
